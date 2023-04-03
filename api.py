from contextlib import ExitStack
import io
import mimetypes
import os
import threading
import time
from urllib.parse import urlsplit, urlparse
import uuid
from fastapi import BackgroundTasks, FastAPI, Request
from webui import models, restore_image, restore_video
import requests
import numpy as np
from PIL import Image
import base64
from os import path as osp
from fastapi.responses import StreamingResponse
import re

class Api:

  def __init__(self, app: FastAPI):
    app.add_api_route("/restore/image", self.restore_image_api, methods=["POST"])
    app.add_api_route("/restore/video", self.restore_video_api, methods=["POST"])
    app.add_api_route("/video-stream/{video_name}", self.video_stream_api, methods=["GET"])

  def block_thread(self):
    while 1:
      time.sleep(0.5)

  def restore_image_api(
    self,
    image: str,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = None,
    outscale: int = 2,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    face_enhance: bool = False,
    fp32: bool = False,
    alpha_upsampler: str = "realesrgan",
    gpu_id: int = None,
  ):
    if image.startswith('http://') or image.startswith('https://'):
      image_buffered = io.BytesIO(requests.get(image).content)
    elif image.startswith('data:image'):
      image_buffered = io.BytesIO(base64.b64decode(image.split(',')[-1]))
    else:
      return {
        "code": 400,
        "message": "Invalid params : image",
      }
    image_array = np.array(Image.open(image_buffered))
    image_output = Image.fromarray(restore_image(image_array, model_name, denoise_strength, outscale, tile, tile_pad, pre_pad, face_enhance, fp32, alpha_upsampler, gpu_id))
    buffered = io.BytesIO()
    image_output.save(buffered, format="PNG")
    image_output_base64 = base64.b64encode(buffered.getvalue()).decode()
    return {
      "data": image_output_base64,
    }

  def restore_video_api(
    self,
    request: Request,
    video: str,
    video_name: str = None,
    video_accept: str = "link",  # or base64
    model_name: str = "realesr-animevideov3",
    denoise_strength: float = None,
    outscale: int = 2,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    face_enhance: bool = False,
    fp32: bool = False,
    alpha_upsampler: str = "realesrgan",
    gpu_id: int = None,
    fps: str = "",
    ffmpeg_bin: str = "ffmpeg",
    extract_frame_first: bool = False,
    num_process_per_gpu: int = 1,
  ):
    VIDEO_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "api_cache/videos")
    if not osp.exists(VIDEO_DIR):
      os.makedirs(VIDEO_DIR)
    is_url = video.startswith('http://') or video.startswith('https://')
    is_base64 = video.startswith('data:video/') or video.startswith('data:application/octet-stream;base64,')

    if(video_name!=None):
      video_name = re.sub(r"[\.\\\/\s\"\']+", "_",video_name)

    if is_url:
      video_name_in_url, ext = osp.splitext(osp.basename(urlsplit(video).path))
      if video_name == None: video_name = video_name_in_url
    elif is_base64:
      ext = ".mp4"
      if video_name == None: video_name = str(uuid.uuid4)
    else:
      return {
        "code": 400,
        "message": "Invalid params : video",
      }

    if video.endswith(".m3u8"):
      ext = ".mp4"
    video_input_path = osp.join(VIDEO_DIR, video_name + ext)

    if osp.exists(video_input_path):
      os.remove(video_input_path)

    def video_handle():
      if is_url:
        if video.endswith(".m3u8"):
          os.system(f'ffmpeg -i {video} -o {video_input_path}')
        else:
          response = requests.get(video, stream=True)
          with open(video_input_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
              if chunk:
                f.write(chunk)
      elif is_base64:
        video_decoded = base64.b64decode(video.split(',')[-1])
        with open(video_input_path, 'wb') as f:
          f.write(video_decoded)
      restore_video(video_input_path, model_name, denoise_strength, outscale, tile, tile_pad, pre_pad, face_enhance, fp32, alpha_upsampler, gpu_id, fps, ffmpeg_bin, extract_frame_first, num_process_per_gpu)

    if video_accept == "base64":
      video_handle()
      with open(video_input_path, 'rb') as f:
        video_bytes = f.read()
      video_output_base64 = base64.b64encode(video_bytes).decode('utf-8')
      return {
        "data": video_output_base64,
      }
    elif video_accept == "link":
      parsed_url = urlparse(str(request.url))
      restore_video_thread = threading.Thread(target=video_handle)
      restore_video_thread.start()
      return {
        "data": f"{parsed_url.scheme}://{parsed_url.netloc}/video-stream/{video_name}_{str(outscale)}x.{model_name}.mp4",  # sync with restore_video().final_path
      }
    else:
      return {
        "code": 400,
        "message": "Invalid params: video_accept",
      }

  def video_stream_api(self, video_name: str, background_tasks: BackgroundTasks):
    VIDEO_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "api_cache/videos")
    video_path = osp.join(VIDEO_DIR, video_name)
    file_like = open(video_path, mode="rb")
    background_tasks.add_task(lambda: file_like.close())
    return StreamingResponse(file_like, media_type=mimetypes.guess_type(video_path)[0])