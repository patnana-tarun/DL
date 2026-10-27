# EdgeVision on NVIDIA Jetson Orin: step-by-step guide

EdgeVision turns a Jetson Orin and a camera into a self-contained crowd-analytics box you can talk to:

- it detects people in every frame with a fine-tuned **YOLO26n** detector (ONNX, run by onnxruntime with
  **TensorRT FP16**),
- follows them with **ByteTrack**,
- computes crowd analytics online: people in view, unique people, line crossings, dwell, zones,
  loitering, and alert rules,
- stores everything in **SQLite**,
- and runs a small vision-language model, **Qwen3-VL-2B** (llama.cpp), that writes scene reports,
  double-checks alerts, and answers your questions. It can also change the alert rules and the counting
  line through tool calls.

You use it from a **web page** (`http://<jetson-ip>:8000`) or from a **terminal chat**. Nothing leaves the
device.

This guide assumes you have **never used a Jetson**. Follow the steps in order. Commands in grey boxes
are typed in a terminal: on the Jetson (directly or over SSH), or on your PC where it says so.

> All performance figures for the Jetson in this guide are **expectations, not measurements**. Measure
> them on your own board with `scripts/04_warmup_detector.sh` and `--report` (step 9).

---

## Contents

- [Architecture](#architecture)
- [Step 0: what you need](#step-0-what-you-need)
- [Step 1: install JetPack 6 and first boot](#step-1-install-jetpack-6-and-first-boot)
- [Step 2: performance mode, swap, monitoring](#step-2-performance-mode-swap-monitoring)
- [Step 3: copy the project from your PC](#step-3-copy-the-project-from-your-pc)
- [Step 4: run the setup scripts](#step-4-run-the-setup-scripts)
- [Step 5: test with a video file and with a stream](#step-5-test-with-a-video-file-and-with-a-stream)
- [Step 6: connect a camera](#step-6-connect-a-camera)
- [Step 7: run the full system, open the UI, chat](#step-7-run-the-full-system-open-the-ui-chat)
- [Step 8: start at boot (systemd)](#step-8-start-at-boot-systemd)
- [Step 9: choosing model sizes per board](#step-9-choosing-model-sizes-per-board)
- [Step 10: troubleshooting](#step-10-troubleshooting)
- [Step 11: how this maps to the research](#step-11-how-this-maps-to-the-research)
- [Folder layout and cheat sheet](#folder-layout-and-cheat-sheet)

---

## Architecture

```
 USB / CSI / IP camera, RTSP stream or video file
                 |
                 v
 ffmpeg (or GStreamer nvarguscamerasrc for CSI)   separate process: decode, scale to 1280x720, raw BGR
                 |
                 v
 +--------------------------- python3 -m edgevision (one process) ----------------------------+
 |  source thread: keeps the newest frame (live) / buffers frames (file)                      |
 |       |                                                                                    |
 |  main loop:  YOLO26n person detector  ->  ByteTrack  ->  crowd analytics + alert rules     |
 |              (onnxruntime, TensorRT       (trackers)      occupancy, unique people, dwell, |
 |               FP16, engine cached in                      line crossings, zones, loitering |
 |               models/trt_cache)                                  |          |              |
 |                                                   SQLite <-------+          |              |
 |                                            data/edgevision.db               v              |
 |  VLM thread: scene reports, alert checks, chat  <----- alert / timer / question            |
 |       |  ^ tool calls: set_alert, remove_alert, set_counting_line, get_history ...         |
 |       |  +-------------------------------------------------> analytics rules               |
 |  web server :8000  (live video, chart, events, chat)   <---- browser / tools/chat_cli.py   |
 +-------|------------------------------------------------------------------------------------+
         v  OpenAI-compatible HTTP API
 llama-server :8080 (llama.cpp, CUDA)  Qwen3-VL-2B-Instruct Q4_K_M + mmproj Q8_0, --jinja (tool calls)
```

Details (threads, API, every config key, latency budget): [`docs/architecture.md`](docs/architecture.md).

---

## Step 0: what you need

| Item | Recommendation |
|---|---|
| Jetson | **Jetson Orin Nano 8 GB Developer Kit** (cheapest that runs everything), **Orin NX 16 GB**, or **AGX Orin 32/64 GB**. Orin Nano 4 GB / Orin NX 8 GB work for the detector, but leave little memory for the VLM (step 9). |
| Storage | **NVMe SSD (M.2 2280, 256 GB or more)** strongly recommended: it is much faster than a microSD card for loading models and building llama.cpp. A microSD card (64 GB or more, UHS-I U3 / A2) works to get started. |
| Power | The power supply that came with the developer kit. Weak supplies cause random reboots under load. |
| Camera (one of) | **USB webcam** (UVC, easiest; 720p or 1080p); **CSI ribbon camera** (IMX219 such as the Raspberry Pi Camera v2, or IMX477; the Orin Nano dev kit needs a 22-pin cable); **IP camera** with an RTSP URL. No camera? Use a video file or a stream from your PC (step 5). |
| Network | Ethernet cable to the same router as your PC (recommended). Wi-Fi works too (the Orin Nano dev kit includes a Wi-Fi card). |
| For setup | A monitor (DisplayPort on the Orin Nano dev kit), USB keyboard and mouse for the first boot. After that everything can be done over SSH. |
| A PC | Linux, Windows or macOS, to flash the SD card / NVMe and to copy this folder. SDK Manager (optional) needs an Ubuntu 20.04/22.04 x86 PC. |

---

## Step 1: install JetPack 6 and first boot

JetPack is NVIDIA's operating system for Jetson (Ubuntu 22.04 + CUDA + cuDNN + TensorRT). This project
targets **JetPack 6.x** (L4T R36), preferably **JetPack 6.2**, which adds the "Super" power modes for
Orin Nano / Orin NX. (Newer JetPack 7.x releases were not tested with this guide.)

### Option A: microSD card image (Orin Nano Developer Kit, easiest)

1. **Firmware first (Orin Nano Developer Kit only).** Kits that left the factory with JetPack 5 firmware
   **cannot boot a JetPack 6 card** until the firmware (QSPI bootloader) is updated. Follow NVIDIA's
   *Initial Setup Guide for Jetson Orin Nano Developer Kit* on jetson-ai-lab.com. In short:
   - Connect a monitor and power on without a card. Read the firmware version on the NVIDIA splash
     screen (or in the UEFI menu, key `Esc`).
   - If it is **older than 36.x**, first boot a **JetPack 5.1.3** SD card image once. It schedules a
     firmware update; let it reboot and finish.
   - Then boot the **JetPack 6.2** SD card. It updates the firmware to 36.4.3 on the next reboot
     (`sudo reboot` once more if it asks).
2. **Flash the card.** On your PC, download the **JetPack 6.2 SD card image** for "Jetson Orin Nano
   Developer Kit" from NVIDIA's JetPack download page. Write it to the microSD card with
   **Balena Etcher** (select the downloaded .zip/.img, select the card, Flash).
3. Insert the card into the slot under the module, connect monitor, keyboard, mouse, Ethernet, and power.

### Option B: NVIDIA SDK Manager (all Orin modules; can install straight to NVMe)

1. On an Ubuntu 20.04/22.04 x86 PC, install **NVIDIA SDK Manager** from developer.nvidia.com (free
   NVIDIA developer account).
2. Put the Jetson into **recovery mode**: power off; on the Orin Nano / NX dev kit place a jumper
   between the **FC REC** and **GND** pins of the button header (under the module), then power on. On the
   AGX Orin dev kit hold the **Force Recovery** button while pressing **Power**. Connect the USB-C port
   to the PC.
3. In SDK Manager pick your board, **JetPack 6.2**, storage **NVMe** (or eMMC/SD), and select both
   "Jetson Linux" and "Jetson Runtime Components". Flash. Remove the jumper afterwards.

### First boot

1. A setup wizard appears: accept the license, choose language, keyboard, time zone, **user name and
   password** (remember them), and the network. If asked for a power mode, choose the highest one
   (step 2 explains it).
2. Open a terminal (`Ctrl+Alt+T`) and update:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo reboot
   ```
3. Find the Jetson's IP address (you will use it from the PC):
   ```bash
   hostname -I          # e.g. 192.168.1.42
   ```
4. From now on you can work from your PC over SSH (enabled by default):
   ```bash
   ssh <user>@<jetson-ip>        # on your PC
   ```

### Optional: put the project on the NVMe SSD when you booted from the SD card

If the system runs from the SD card but an NVMe SSD is installed, use the SSD for the project folder.
**The format command erases the SSD.** Check the device name with `lsblk` first.

```bash
lsblk                                   # find the SSD, usually nvme0n1
sudo mkfs.ext4 /dev/nvme0n1             # ERASES the SSD
sudo mkdir -p /mnt/nvme
echo "UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1) /mnt/nvme ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount -a && sudo chown $USER:$USER /mnt/nvme
ln -s /mnt/nvme/edge_deploy ~/edge_deploy    # so ~/edge_deploy points to the SSD (step 3 copies into it)
```

---

## Step 2: performance mode, swap, monitoring

**Power mode.** Jetsons start in a power-saving mode. List the modes your board offers and pick the
fastest one:

```bash
sudo nvpmodel -q                               # current mode
grep -E "POWER_MODEL ID" /etc/nvpmodel.conf    # all modes of this board (ID and name)
```

| Board (JetPack 6.2) | Fastest mode | Command |
|---|---|---|
| Orin Nano 8 GB (Super) | MAXN SUPER | `sudo nvpmodel -m 2` |
| Orin Nano 4 GB, Orin NX 16 GB / 8 GB (Super) | MAXN SUPER | check the list above, then `sudo nvpmodel -m <id>` |
| AGX Orin | MAXN | `sudo nvpmodel -m 0` |
| any board, JetPack 6.0/6.1 | highest wattage in the list | `sudo nvpmodel -m <id>` |

You can also change it from the power icon in the desktop's top bar. The board may ask to reboot.
Then lock the clocks at their maximum for the current mode (this is reset at every reboot; step 8 shows
how to make it permanent):

```bash
sudo jetson_clocks
```

MAXN modes draw more power and get warm: keep the fan connected and the heatsink free.

**Swap.** CPU and GPU share the same RAM on a Jetson. JetPack enables compressed RAM swap (zram) by
default. Add a real swap file on the SSD as a safety net, especially for the llama.cpp build:

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h                                # "Swap:" should now show about 8G more
```

Optional on the 8 GB Orin Nano: once everything works over SSH, boot without the desktop to free
memory (roughly several hundred MB; check with `free -h`):

```bash
sudo systemctl set-default multi-user.target     # undo: sudo systemctl set-default graphical.target
```

**Monitoring with jtop** (CPU/GPU load, memory, temperature, power mode):

```bash
sudo pip3 install -U jetson-stats
sudo reboot            # or: sudo systemctl restart jtop.service
jtop                   # q to quit
```

---

## Step 3: copy the project from your PC

The `edge_deploy/` folder is self-contained: code, configs, detector models, scripts, and the Qwen3-VL
GGUF files if you already downloaded them on the PC. Two things in it are **x86 desktop binaries that do
not run on the Jetson** and must be left out: `third_party/llama.cpp-vulkan-x64/` and `tools/bin/mediamtx`.
Also leave out `.venv/`, the TensorRT cache (`models/trt_cache/`, engines are built per device) and `data/`
(desktop database and logs), as well as `results/` if it exists (desktop evaluation runs).

**From the PC (Linux / macOS / WSL), in the folder that contains `edge_deploy/`:**

```bash
rsync -avP \
  --exclude '.venv/' --exclude '__pycache__/' \
  --exclude 'third_party/llama.cpp-vulkan-x64/' --exclude 'third_party/llama.cpp/' \
  --exclude 'tools/bin/mediamtx' --exclude 'models/trt_cache/' --exclude 'data/' --exclude 'results/' \
  edge_deploy/ <user>@<jetson-ip>:~/edge_deploy/
```

The trailing slashes matter (`edge_deploy/` -> `~/edge_deploy/`). Re-running the command only copies
changed files. Without rsync (e.g. plain Windows), use tar over ssh, or copy the folder with a USB stick
and delete the two x86 folders/files on the Jetson:

```bash
tar --exclude='.venv' --exclude='__pycache__' --exclude='third_party/llama.cpp-vulkan-x64' \
    --exclude='tools/bin/mediamtx' --exclude='models/trt_cache' --exclude='data' --exclude='edge_deploy/results' \
    -czf - edge_deploy | ssh <user>@<jetson-ip> 'tar -xzf - -C ~'
```

**Copy a test video** as well (used in step 5; `aa1.mp4` is the 1280x720 clip from the study):

```bash
ssh <user>@<jetson-ip> 'mkdir -p ~/edge_deploy/data'
rsync -avP videos/aa1.mp4 <user>@<jetson-ip>:~/edge_deploy/data/      # same folder as above (the one containing videos/ and edge_deploy/)
```

From here on, all commands run **on the Jetson inside `~/edge_deploy`**:

```bash
cd ~/edge_deploy
```

---

## Step 4: run the setup scripts

Run them one after the other. Every script prints what it does, stops with a clear message if something
is wrong, and is safe to run again (it skips work already done).

| # | Script | What it does | Time (expectation) |
|---|---|---|---|
| 0 | `scripts/00_check_system.sh` | Read-only check: JetPack/L4T version, CUDA, cuDNN, TensorRT, Python, RAM, swap, disk, power mode, cameras, ports. | seconds |
| 1 | `scripts/01_install_deps.sh` | apt packages (ffmpeg, GStreamer, v4l-utils, cmake, compilers ...), `nvidia-jetpack` if missing, the Python venv `.venv` with `requirements-jetson.txt`, and **onnxruntime-gpu** from the Jetson AI Lab index. Checks that `TensorrtExecutionProvider` is available. | 5-20 min |
| 2 | `scripts/02_build_llamacpp.sh` | Clones llama.cpp into `third_party/llama.cpp` and builds `llama-server` + `llama-mtmd-cli` with CUDA for Orin (sm_87). | 20-60 min |
| 3 | `scripts/03_download_models.sh` | Downloads Qwen3-VL-2B (GGUF Q4_K_M + mmproj Q8_0, ~1.5 GB) into `models/vlm/` (skipped if copied in step 3) and MediaMTX arm64 into `tools/bin/`. `VLM_SIZE=4B` for the 4B model. | depends on network |
| 4 | `scripts/04_warmup_detector.sh` | Builds the TensorRT engine for the detector (first time only), caches it in `models/trt_cache/`, prints the provider and the median latency on your board. | 2-10 min first time |
| 5 | `scripts/05_run.sh` | Starts the VLM server if needed, waits until it answers, runs the pipeline. Ctrl-C stops both. | - |

```bash
cd ~/edge_deploy
chmod +x scripts/*.sh          # only needed if the copy lost the execute permission
scripts/00_check_system.sh
scripts/01_install_deps.sh     # asks for your sudo password
scripts/02_build_llamacpp.sh
scripts/03_download_models.sh
scripts/04_warmup_detector.sh
```

What a good result looks like:

- `01` ends with `OK: TensorrtExecutionProvider is available.`
- `02` ends with `OK: third_party/llama.cpp/build/bin/llama-server`.
- `04` prints `provider : TensorrtExecutionProvider` and a latency line; write the number down.

Notes:
- `01` uses `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/` for onnxruntime-gpu. That index matches
  **JetPack 6.1/6.2 (CUDA 12.6)**. On JetPack 6.0 (CUDA 12.2) upgrade to 6.2, or set
  `JETSON_PIP_INDEX` to a matching index.
- The venv uses `--system-site-packages`, and installs numpy 2.x and opencv-python inside it (required
  by `trackers==2.6.1`). They take precedence over JetPack's apt numpy/OpenCV. Always run the project with
  `.venv/bin/python` (the scripts do this for you).
- If you already have a desktop with the project, the `.gguf` files can be copied instead of downloaded.

---

## Step 5: test with a video file and with a stream

Do these tests before connecting a camera. They show that the detector, tracker, VLM and UI work.

### 5a. A video file

```bash
cd ~/edge_deploy
# quick headless benchmark, no VLM, no UI: 900 frames, writes a report
.venv/bin/python -m edgevision --config configs/jetson.yaml \
    --set source.uri=data/aa1.mp4 --set vlm.enabled=false --set server.enabled=false \
    --max-frames 900 --report data/report_file.json
```

The last log line is the run report: `throughput_fps`, `latency_ms.detect_ms.p50`, `unique_people`, ...
(`configs/jetson.yaml` resamples to 15 fps, so 900 frames = 60 s of video).

Now the full system on the file, played at real speed in a loop, as if it were a camera:

```bash
scripts/05_run.sh --set source.uri=data/aa1.mp4 --set source.realtime=true --set source.loop=true
```

Open `http://<jetson-ip>:8000` on your PC (step 7). Stop with `Ctrl+C`.

### 5b. A live RTSP stream from your PC over the network

This is the closest thing to an IP camera. On the **PC** (in its `edge_deploy` folder), publish the video
with MediaMTX + ffmpeg (both available on the desktop setup):

```bash
python3 tools/stream_videos.py ../videos/aa1.mp4       # prints: cam1: ../videos/aa1.mp4 -> rtsp://127.0.0.1:8554/cam1
```

The PC now serves `rtsp://<pc-ip>:8554/cam1` to the network (allow TCP port 8554 in the PC firewall,
e.g. `sudo ufw allow 8554/tcp`). On the **Jetson**:

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height rtsp://<pc-ip>:8554/cam1   # should show codec_name=h264, width=1280, height=720
scripts/05_run.sh --set source.uri=rtsp://<pc-ip>:8554/cam1
```

Other protocols: `python3 tools/stream_videos.py ../videos/aa1.mp4 --protocol udp --host <jetson-ip> --port 5000`
on the PC, and `--set source.uri=udp://0.0.0.0:5000` on the Jetson (no server needed). `--reencode
--scale 1280:720` re-encodes a large video to a Jetson-friendly 720p H.264 stream.

### 5c. A stream on the Jetson itself

`03_download_models.sh` installed MediaMTX arm64 in `tools/bin/`. In a second terminal (or SSH session):

```bash
cd ~/edge_deploy
python3 tools/stream_videos.py data/aa1.mp4            # starts tools/bin/mediamtx with tools/mediamtx.yml
```

and in the first one:

```bash
scripts/05_run.sh --set source.uri=rtsp://127.0.0.1:8554/cam1
```

Live sources keep only the newest frame: if the Jetson cannot keep up, frames are dropped (shown as
`dropped` in the UI) instead of building up delay.

---

## Step 6: connect a camera

Pick one camera and set `source.uri` in `configs/jetson.yaml` (edit with `nano configs/jetson.yaml`),
or pass `--set source.uri=...` on the command line. The default in the file is the USB camera `/dev/video0`.

### USB camera

```bash
v4l2-ctl --list-devices                         # which /dev/videoN belongs to the camera
v4l2-ctl -d /dev/video0 --list-formats-ext      # supported formats, sizes and frame rates
```

Use a size and frame rate from the list. Most webcams deliver 1280x720 at 30 fps only as **MJPG**;
then tell ffmpeg to ask for MJPG. Test one frame first:

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -i /dev/video0 -frames:v 1 -y data/usb_test.jpg
```

and in `configs/jetson.yaml`:

```yaml
source:
  uri: /dev/video0
  width: 1280            # must be a size the camera supports (EdgeVision passes it to ffmpeg as -video_size)
  height: 720
  fps: 15
  input_args: ["-input_format", "mjpeg"]     # only if the camera needs MJPG for this size
```

If you get "Permission denied", add yourself to the `video` group (`sudo usermod -aG video $USER`, then log
out and in). "Device or resource busy" means another program uses the camera (step 10).

### CSI ribbon camera (IMX219 / IMX477)

1. **Power off**, connect the ribbon cable (contacts facing the right way, see the camera's guide),
   power on.
2. Enable the camera in the device tree once:
   ```bash
   sudo /opt/nvidia/jetson-io/jetson-io.py
   ```
   "Configure Jetson 24pin CSI Connector" (wording varies by board) -> "Configure for compatible
   hardware" -> choose your camera (e.g. "Camera IMX219 Dual") -> Save and reboot.
3. Test. With a monitor attached:
   ```bash
   nvgstcapture-1.0                     # preview window; q + Enter to quit
   # or
   gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! nvvidconv ! xvimagesink
   ```
   Over SSH (no monitor), save a few frames instead:
   ```bash
   gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 num-buffers=30 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' \
       ! nvvidconv ! 'video/x-raw,format=I420' ! jpegenc ! multifilesink location=data/csi_%02d.jpg
   ```
4. Config: `uri: csi://0` (second camera: `csi://1`), keep `width: 1280`, `height: 720`, `fps: 15`.
   EdgeVision runs `gst-launch-1.0 nvarguscamerasrc` itself.

If it fails with "Failed to create CaptureSession", restart the camera daemon:
`sudo systemctl restart nvargus-daemon`.

### IP camera (RTSP)

Find the RTSP URL in the camera's manual or web page. Use the **sub-stream** (720p, H.264) when there is
one: by default the Jetson decodes with ffmpeg on the CPU, and 1080p/4K H.265 main streams cost much more
CPU. For H.264 IP cameras you can use the Jetson's **hardware decoder** instead:
`--set source.decoder=gstreamer` (pipeline `rtspsrc ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv`;
it needs the GStreamer plugins installed by `01_install_deps.sh`).
Typical patterns (check your model):

| Brand | URL (sub-stream) |
|---|---|
| Hikvision | `rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102` |
| Dahua / Amcrest | `rtsp://user:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1` |
| Reolink | `rtsp://user:pass@192.168.1.100:554/h264Preview_01_sub` |

Special characters in the password must be URL-encoded (`@` -> `%40`, `:` -> `%3A`, `#` -> `%23`). Test:

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"
```

then set `uri:` to that URL (keep it in quotes in YAML if it contains `&` or `?`).

---

## Step 7: run the full system, open the UI, chat

```bash
cd ~/edge_deploy
scripts/05_run.sh                       # uses configs/jetson.yaml
```

The script starts the VLM server (log: `data/llama_server.log`), waits until `http://127.0.0.1:8080/v1/models`
answers, and starts the pipeline. You should see lines like `UI on http://localhost:8000`,
`VLM endpoint ...: online`, and every 300 frames a status line with fps, occupancy and detector latency.

Useful variants:

```bash
scripts/05_run.sh --set source.uri=csi://0                 # any config key can be overridden
CONFIG=configs/my_site.yaml scripts/05_run.sh              # your own copy of the config
CTX=4096 scripts/05_run.sh                                 # smaller VLM context (less memory)
VLM_ARGS="-np 1" scripts/05_run.sh                         # extra llama-server flags (one request slot)
.venv/bin/python -m edgevision --config configs/jetson.yaml --set vlm.enabled=false   # no VLM at all
```

### The web UI

On any computer in the same network open **`http://<jetson-ip>:8000`**. You get:

- the live annotated video (boxes with track IDs, counting line, crossing counts, status bar),
- a chart of people in view over the last 10 minutes,
- measurements (in view, unique people, crossings, dwell, crowd level, trend),
- **Ask the camera**: the chat box, with quick buttons,
- VLM scene reports (every `vlm.observe_every_s` = 120 s), alert rules, and the event list.

- a foot-traffic heatmap (where people walked), refreshed every 30 s.

By default the UI has no login: anyone who can reach port 8000 is treated as an *operator* (they can watch
and change alert rules). On any shared network give every person their own token and role:

```yaml
server:
  users:
    - {name: alice, token: "a-long-random-secret", role: admin}     # may also remove ALL alerts
    - {name: bob,   token: "another-long-secret",  role: operator}  # may change rules and the line
    - {name: guest, token: "a-third-secret",       role: viewer}    # may watch and ask, not change
```

and open `http://<jetson-ip>:8000/?token=<your token>`; `tools/chat_cli.py` and other API clients send the
same value as `?token=` or the `X-Token` header. (The older single `server.token` still works and means admin.)

**Policy layer** (`edgevision/policy.py`, settings under `policy:`): the VLM only *proposes* changes. Every
change, whether it comes from the model, the fixed rules or a `/command`, is checked first: your role, valid
values (no "alert above a million people"), that you actually asked for it in this message, a rate limit, and
for removing or weakening an alert from the chat a confirmation (`/confirm <id>`, shown in the answer;
`/pending` lists them, `/cancel <id>` drops one). Every decision goes into a tamper-evident audit log:
`/audit` shows the last entries and whether the hash chain is intact. Data older than `store.retention_h`
(72 h) is deleted automatically, except the audit log.

**How reliable is the chat?** The VLM is a 2-billion-parameter model. Counts and alerts always come from the
detector, never from the VLM. When an alert fires the VLM *describes* the scene (it does not confirm or reject
the alert: in tests it could not). Requests that change settings go through tools; obvious requests ("remove
all alerts", "make the line horizontal") are also handled by fixed rules if the model gets them wrong, and
every such answer ends with a `[system] Changed: ...` line that states what really changed. When in doubt,
use the `/commands` below: they never depend on the model.

### Chat from a terminal

```bash
.venv/bin/python tools/chat_cli.py                                  # on the Jetson
python3 tools/chat_cli.py --url http://<jetson-ip>:8000              # from the PC (stdlib only)
python3 tools/chat_cli.py --url http://<jetson-ip>:8000 -q "how many people are in view?"
```

### What to ask

Questions in plain language go to Qwen3-VL together with the current frame and the live measurements.
Counts come from the tracker (more accurate than a small VLM counting heads); the VLM adds what the
numbers cannot say.

| You type | What happens |
|---|---|
| `What is happening in the scene right now?` | short description of the frame and activities |
| `Is anything unusual or unsafe?` | the VLM looks for anomalies |
| `How many people crossed the line in the last 10 minutes?` | the VLM calls `get_history` |
| `Show me the recent alerts.` | `get_recent_events(kind=alert)` |
| `Alert me if more than 10 people are in view for 5 seconds.` | `set_alert(occupancy > 10, for_s=5)`; the rule appears in the UI |
| `Count people crossing a horizontal line in the middle of the image.` | `set_counting_line(0, 0.5, 1, 0.5)` |
| `Remove all alerts.` | `remove_alert("all")` |
| `Is anyone loitering?` | answer from measurements + frame |

Commands starting with `/` are handled directly by the pipeline. They are instant, deterministic, and
work even when the VLM is offline:

| Command | Effect |
|---|---|
| `/help` | list of commands |
| `/stats` | live measurements |
| `/history 30` | occupancy summary and event counts of the last 30 minutes |
| `/events 10` | events of the last 10 minutes |
| `/alerts` | current alert rules |
| `/alert occupancy > 10 for 5` | add a rule (metrics: `occupancy`, `unique_people`, `crossings_total`, `crossings_a_to_b`, `crossings_b_to_a`, `crossings_last_min`, `max_dwell_s`, `loitering`, `zone:<name>`) |
| `/unalert crowding` | remove a rule by name (`/unalert all` removes all; admin role) |
| `/line 0 0.5 1 0.5` | move the counting line (x1 y1 x2 y2, 0-1 image coordinates) |
| `/observe` | ask the VLM for a scene report now |
| `/pending`, `/confirm c3`, `/cancel c3` | changes from chat that wait for your confirmation |
| `/audit` | last audit-log entries and whether the hash chain verifies |

Rule changes made in the chat apply from the next frame on, but are not written back to the YAML file.
Put permanent rules under `alerts:` in the config.

---

## Step 8: start at boot (systemd)

When everything works with `scripts/05_run.sh`, install two system services: the VLM server and the
pipeline. The pipeline service starts the VLM service too.

```bash
cd ~/edge_deploy
scripts/04_warmup_detector.sh             # make sure the TensorRT engine is cached first
scripts/install_service.sh                # CONFIG=..., VLM_SIZE=4B, CTX=4096 are accepted
```

This writes `/etc/systemd/system/edgevision-vlm.service` and `/etc/systemd/system/edgevision.service`
(from the templates in `scripts/systemd/`), enables them, and starts them. They run as your user with
`WorkingDirectory` = this folder and `Restart=on-failure`.

```bash
systemctl status edgevision edgevision-vlm
journalctl -u edgevision -f               # live pipeline log (Ctrl+C to leave)
journalctl -u edgevision-vlm -f           # VLM server log
sudo systemctl restart edgevision         # after editing configs/jetson.yaml
sudo systemctl stop edgevision edgevision-vlm
scripts/install_service.sh --disable      # stop and do not start at boot any more
```

`NO_VLM=1 scripts/install_service.sh` installs only the pipeline (also set `vlm.enabled: false`).
Do not run `scripts/05_run.sh` while the services are running: the ports would be taken.

**Max clocks at boot (optional).** `nvpmodel` remembers the mode, but `jetson_clocks` does not. To apply it
at every boot:

```bash
sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'EOF'
[Unit]
Description=Lock Jetson clocks at maximum
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now jetson-clocks.service
```

---

## Step 9: choosing model sizes per board

### Detector input size

Three exports of the same fine-tuned YOLO26n are in `models/`. Accuracy below is **measured** on the
study's test set (`models/selected_model.json`, desktop evaluation of the ONNX files). Speed on a Jetson is
not measured here.

| File | Input | AP50-95 | AP50 | AP small | Use on |
|---|---|---|---|---|---|
| `yolo26n_person_736x1280.onnx` | 736x1280 | 0.827 | 0.963 | 0.552 | Orin NX 16 GB, AGX Orin (study resolution, best for small/distant people) |
| `yolo26n_person_544x960.onnx` | 544x960 | 0.794 | 0.959 | 0.505 | **Orin Nano 8 GB (default in `configs/jetson.yaml`)** |
| `yolo26n_person_384x640.onnx` | 384x640 | 0.682 | 0.925 | 0.295 | Orin Nano 4 GB, or when people are large / close |

Switch with `detector.weights` in the config (or `--set detector.weights=models/yolo26n_person_736x1280.onnx`).
Each file gets its own TensorRT engine: run `scripts/04_warmup_detector.sh` again after switching.

Rule of thumb: at `source.fps: 15` the whole loop must take less than 66 ms per frame. Look at
`detect_ms` and `loop_ms` in `/api/state` (or a `--report`). If `dropped` keeps growing, use a smaller
input or a lower `source.fps`. If there is plenty of headroom, try the larger model.

### VLM: Qwen3-VL-2B or 4B

| | Qwen3-VL-2B-Instruct (default) | Qwen3-VL-4B-Instruct |
|---|---|---|
| Files | `Qwen3VL-2B-Instruct-Q4_K_M.gguf` (1.11 GB) + `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` (0.45 GB) | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` (2.50 GB) + `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` (0.45 GB) |
| Boards | all 8 GB+ boards | Orin NX 16 GB, AGX Orin |
| Download | `scripts/03_download_models.sh` | `VLM_SIZE=4B scripts/03_download_models.sh` |
| Start | `scripts/05_run.sh` | `VLM_SIZE=4B scripts/05_run.sh` and `vlm.model: qwen3-vl-4b` |

### Memory budget (approximate)

CPU and GPU share the RAM. File sizes are exact; the KV-cache figures are computed from the model
architecture (f16 cache: about 0.11 MB per token for 2B, 0.14 MB per token for 4B); everything else is a
**rough estimate** to be checked with `jtop` on your board.

| Item | Orin Nano 8 GB, 2B, CTX 4096 | 2B, CTX 8192 (default) | Orin NX 16 / AGX, 4B, CTX 8192 |
|---|---|---|---|
| JetPack, desktop, system services | ~1-2 GB (less without the desktop) | ~1-2 GB | ~1-2 GB |
| EdgeVision process (Python, onnxruntime, TensorRT engine, CUDA context) | ~1-1.5 GB | ~1-1.5 GB | ~1-1.5 GB |
| VLM weights + vision projector | 1.55 GB | 1.55 GB | 2.95 GB |
| VLM KV cache | ~0.45 GB | ~0.9 GB | ~1.2 GB |
| llama.cpp compute buffers (text + vision encoder) | ~0.3-0.8 GB | ~0.3-0.8 GB | ~0.3-0.8 GB |
| **Total (estimate)** | **~4.3-6.3 GB** | **~4.8-6.8 GB** | **~6.5-8.5 GB** |

On the 8 GB Orin Nano the 2B model fits. `CTX=4096` (one question + image + context fits easily; the
pipeline sends one VLM request at a time) and `VLM_ARGS="-np 1"` leave more margin. The 4B model does not
fit comfortably on 8 GB. On 4 GB boards set `vlm.enabled: false` or point `vlm.endpoint` to a VLM server
on another machine.

### Speed expectations (not measured; verify on your board)

- **Detector**: on an Orin Nano in MAXN SUPER with TensorRT FP16 at 544x960, the detector is expected to
  stay well inside the 66 ms budget of 15 fps. `scripts/04_warmup_detector.sh` prints the real number.
- **VLM**: with Qwen3-VL-2B on an Orin Nano, expect a chat answer to take several seconds to a few tens
  of seconds (image encoding + prompt + generated text). `vlm.image_max_side: 768` and short questions
  make it faster. The UI shows `last_latency_s`; `data/llama_server.log` prints tokens per second.
- While the VLM is working it shares the GPU with the detector, so `detect_ms` is expected to rise
  temporarily.
- **Measure it**: `python tools/bench.py --config configs/jetson.yaml --video ../videos/aa1.mp4 --duration 1800`
  runs detector-only, VLM-only and both together for 30 min each and writes latency percentiles, dropped
  frames, power (`tegrastats`), temperatures and memory to `results/bench_<host>_<time>.json`. Please send
  that file back: it is the Jetson measurement the paper still needs.

---

## Step 10: troubleshooting

| Problem | Cause and fix |
|---|---|
| First start "hangs" for minutes after `onnxruntime ... providers` | TensorRT is building the engine (first run per model and board). Wait; it is cached in `models/trt_cache/`. Run `scripts/04_warmup_detector.sh` once so services never do this at boot. If you change JetPack/TensorRT, the engine is rebuilt automatically (or move the old `models/trt_cache` folder away). |
| Log shows `providers ['CPUExecutionProvider']` or detection is very slow | onnxruntime cannot use TensorRT/CUDA, so it fell back to CPU. Check: `.venv/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"`. Fix: `FORCE_ORT=1 scripts/01_install_deps.sh` (removes the CPU `onnxruntime` package, installs `onnxruntime-gpu` from the Jetson index). Make sure `nvidia-jetpack` is installed (`dpkg -l nvidia-jetpack`) and the index matches your CUDA (`nvcc --version`, JetPack 6.1/6.2 = 12.6). |
| `libcudnn.so.9` / `libnvinfer.so.10` not found | TensorRT/cuDNN from JetPack missing: `sudo apt install nvidia-jetpack`. |
| VLM server exits, or "out of memory" / system freezes | Too little RAM. Use `CTX=4096` (or 2048) and `VLM_ARGS="-np 1"`, keep the 2B model, set `vlm.image_max_side: 640`, close the desktop (step 2), add swap, use the 544x960 or 384x640 detector. Watch memory in `jtop`. |
| `llama-server not found` | Run `scripts/02_build_llamacpp.sh`. Do not use `third_party/llama.cpp-vulkan-x64` on the Jetson (x86 build). |
| llama.cpp build killed / `nvcc` crashes | Out of memory while compiling: `JOBS=2 scripts/02_build_llamacpp.sh` and add swap (step 2). `nvcc not found`: `sudo apt install nvidia-jetpack`. |
| llama-server rejects an option after a llama.cpp update | Options change upstream. Build a known version: `LLAMA_REF=<tag> REBUILD=1 scripts/02_build_llamacpp.sh` (tags on github.com/ggml-org/llama.cpp/tags), or adjust `scripts/start_vlm.sh`. |
| UI says VLM offline / chat says "VLM error" | `curl http://127.0.0.1:8080/v1/models` on the Jetson. Look at `data/llama_server.log` (or `journalctl -u edgevision-vlm`). The `/commands` still work meanwhile. |
| Answers are slow | Expected for a VLM on an edge board. Lower `vlm.image_max_side`, `vlm.max_tokens`, raise `vlm.observe_every_s` so periodic reports do not queue in front of your question, use MAXN + `jetson_clocks`. |
| USB camera: `Device or resource busy` | Another program has it open (a second EdgeVision, the systemd service, cheese, a browser tab). `sudo fuser -v /dev/video0` shows who. |
| USB camera: black / wrong size / 5 fps | The format is not supported: check `v4l2-ctl --list-formats-ext`, use a listed size, add `input_args: ["-input_format", "mjpeg"]`. |
| CSI camera: `Failed to create CaptureSession` | `sudo systemctl restart nvargus-daemon`; check the cable orientation; run `jetson-io.py` again (step 6). |
| RTSP: high delay | EdgeVision itself keeps only the newest frame. Delay usually comes from the camera's encoder/GOP or from buffering: use the sub-stream, a short keyframe interval, TCP (default `source.rtsp_transport: tcp`), and wired Ethernet. On the PC streamer use `--reencode` (zerolatency, no B-frames). |
| RTSP: `Connection refused` / timeouts | Wrong URL or port, the camera is on another subnet, or a firewall. Test with `ffprobe` (step 6). |
| Cannot open `http://<jetson-ip>:8000` from the PC | Check the pipeline is running (`ss -ltnp | grep 8000`), and the firewall. Ubuntu on Jetson has no firewall active by default; if you enabled `ufw`: `sudo ufw allow 8000/tcp` (UI), `sudo ufw allow 8554/tcp` (only if other machines publish/read RTSP on the Jetson). Port 8080 (VLM) listens on 127.0.0.1 only and does not need to be opened. On the PC streaming in step 5b, allow 8554/tcp. |
| `Address already in use` on 8000/8080/8554 | Something already runs there (often the systemd services): `ss -ltnp | grep -E '8000|8080|8554'`; stop it or change `server.port` / `PORT=`. |
| `numpy` / `cv2` import errors | Run with `.venv/bin/python`, not the system `python3`. If a ROS install sets `PYTHONPATH`, the scripts unset it; do the same by hand: `env -u PYTHONPATH .venv/bin/python ...`. |
| Board reboots under load | Power supply too weak, or overheating: use the original PSU, check the fan, try a lower `nvpmodel` mode. |

Collect information for a bug report with `scripts/00_check_system.sh > data/system.txt` and the run log.

---

## Step 11: how this maps to the research

The deployment is the last step of the study in this repository (`../research/`, results in `../results/`):

1. **Model selection** (`../research/select_model.py` -> `../results/model_selection.json`). YOLO26n and
   RF-DETR Nano were trained on the same data and compared with a weighted decision matrix:

   | Criterion (weight) | YOLO26n | RF-DETR Nano |
   |---|---|---|
   | test AP50-95 (0.30) | 0.825 | 0.725 |
   | label-bias minimum AP (0.10) | 0.726 | 0.630 |
   | tracking mean HOTA (0.25) | 0.702 (10/12 wins) | 0.688 (2/12) |
   | latency, desktop RTX 4060 Laptop GPU (0.20) | 3.68 ms | 7.89 ms |
   | GFLOPs (0.10) | 13.6 | 97.3 |
   | parameters (0.05) | 2.5 M | 30.5 M |
   | **weighted score** | **1.00** | 0.71 |

   YOLO26n wins under every alternative weighting tested (accuracy only, speed only, tracking only, equal,
   bias-aware): `winner_robust: true`. It is also the better fit for an edge GPU (7x fewer FLOPs, 12x fewer
   parameters).
2. **Export** - the same script exported the three ONNX files in `models/` and checked them: the
   736x1280 ONNX scores AP50-95 0.8273 vs 0.8254 for the PyTorch model (difference 0.0019, parity
   passed). The exported head is one-to-many + NMS, because the NMS-free head lost about 0.02 AP in
   crowds (`../results/latency.json`).
3. **Tracking** - the `tracker:` parameters (ByteTrack thresholds, buffer, minimum IoU) are the values
   tuned in `../research/track.py`, copied unchanged into `edgevision/config.py` and
   `models/selected_model.json`; `edgevision/tracker.py` applies the same pre-filter as the study.
4. **Analytics** - the definitions (unique people = tracks of at least 0.25 s, dwell, line crossings with
   hysteresis) match `../research/analytics.py`.
5. **Edge loop** - this folder adds what the offline study cannot: stream decoding with FFmpeg/GStreamer,
   online analytics, persistence, and the VLM in a closed loop (observe -> verify alerts -> chat with tool
   calls that change the analytics rules).

Performance on Jetson hardware is not part of the measured results. To add it, run
`scripts/04_warmup_detector.sh` and a `--report` run (step 5a) on the board and record the board, JetPack
version, power mode and model input size with the numbers.

---

## Folder layout and cheat sheet

```
edge_deploy/
  edgevision/            the application (python -m edgevision)
    config.py            defaults + YAML + --set overrides      stream.py    ffmpeg / GStreamer in and out
    detector.py          ONNX / TensorRT person detector         tracker.py   ByteTrack
    analytics.py         counts, dwell, line, zones, alerts      store.py     SQLite
    vlm.py               Qwen3-VL agent (observe/verify/chat)    commands.py  /commands
    server.py + ui/      web UI and JSON API on :8000            pipeline.py  main loop
  configs/               jetson.yaml (Jetson), desktop.yaml (PC)
  models/                yolo26n_person_*.onnx, selected_model.json, vlm/*.gguf, trt_cache/ (created)
  scripts/               00..05 setup + run, start_vlm.sh, install_service.sh, systemd/ unit templates
  tools/                 stream_videos.py, chat_cli.py, mediamtx.yml, bin/mediamtx
  docs/architecture.md   threads, API, config reference, latency budget
  tests/                 pytest suite (on the PC: .venv/bin/python -m pytest tests)
  data/                  edgevision.db (SQLite), llama_server.log (created at run time)
  requirements-jetson.txt / requirements-desktop.txt
  Dockerfile.jetson      optional container build (not the main path)
```

| I want to... | Command |
|---|---|
| run everything | `scripts/05_run.sh` |
| run on a file | `scripts/05_run.sh --set source.uri=data/aa1.mp4 --set source.realtime=true --set source.loop=true` |
| run without the VLM | `.venv/bin/python -m edgevision --config configs/jetson.yaml --set vlm.enabled=false` |
| only the VLM server | `scripts/start_vlm.sh` (`VLM_SIZE=4B`, `CTX=4096`, `PORT=8081` accepted) |
| benchmark | `.venv/bin/python -m edgevision --config configs/jetson.yaml --set source.uri=data/aa1.mp4 --set server.enabled=false --max-frames 1500 --report data/report.json` |
| chat in the terminal | `.venv/bin/python tools/chat_cli.py` |
| watch the board | `jtop` |
| services | `scripts/install_service.sh`, `journalctl -u edgevision -f` |
| look at the database | `sqlite3 data/edgevision.db 'select kind, count(*) from events group by kind'` (`sudo apt install sqlite3`) |

**Optional Docker path.** `Dockerfile.jetson` builds an image on top of `nvcr.io/nvidia/l4t-jetpack:r36.4.0`
with the Python dependencies, onnxruntime-gpu and llama.cpp. The build and run commands are in the
comments at the top of the file. The native install above is the main, documented path; the container
was not tested on hardware.
