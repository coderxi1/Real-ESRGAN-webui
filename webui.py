import mimetypes
import gradio as gr
import cv2
import os
import shutil
import inference_realesrgan_video as irv
from os import path as osp
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class Struct(dict):
  def __init__(self, **entries):
    entries = {k: v for k, v in entries.items() if k != "items"}
    dict.__init__(self, entries)
    self.__dict__.update(entries)
  def __setattr__(self, attr, value):
    self.__dict__[attr] = value
    self[attr] = value
def structify(o):
    if isinstance(o, list):
        return [structify(o[i]) for i in range(len(o))]
    elif isinstance(o, dict):
        return Struct(**{k: structify(v) for k, v in o.items()})
    return o

models = {
  # model_name : model, netscale, file_url
  "RealESRGAN_x4plus": lambda: (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4, ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]),
  "RealESRNet_x4plus": lambda: (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4, ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"]),
  "RealESRGAN_x4plus_anime_6B": lambda: (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4, ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"]),
  "RealESRGAN_x2plus": lambda: (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2, ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]),
  "realesr-animevideov3": lambda: (SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"), 4, ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"]),
  "realesr-general-x4v3": lambda: (SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"), 4, ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"]),
}

def restore_image(img, model_name, denoise_strength, outscale, tile, tile_pad, pre_pad, face_enhance, fp32, alpha_upsampler, gpu_id):
  output = None
  model, netscale, file_url = models[model_name]()
  model_path = os.path.join('Real-ESRGAN','weights', model_name + '.pth')
  if not os.path.isfile(model_path):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    for url in file_url:
      # model_path will be updated
      model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'Real-ESRGAN', 'weights'), progress=True, file_name=None)

  # use dni to control the denoise strength
  dni_weight = None
  if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
    wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    model_path = [model_path, wdn_model_path]
    dni_weight = [denoise_strength, 1 - denoise_strength]

  # restorer
  upsampler = RealESRGANer(scale=netscale, model_path=model_path, dni_weight=dni_weight, model=model, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=not fp32, gpu_id=gpu_id)
  if face_enhance:  # Use GFPGAN for face enhancement
    from gfpgan import GFPGANer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)
  try:
    if face_enhance:
      _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
      output, _ = upsampler.enhance(img, outscale=outscale,alpha_upsampler=alpha_upsampler)
  except RuntimeError as error:
    print('Error', error)
    print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
  return output

def restore_video(video_path, model_name, denoise_strength, outscale, tile, tile_pad, pre_pad, face_enhance, fp32, alpha_upsampler, gpu_id, fps, ffmpeg_bin, extract_frame_first, num_process_per_gpu):
  output_dir = osp.dirname(video_path)
  video_name, ext = osp.basename(video_path).split('.',2)
  suffix = str(outscale) + "x." + "fps." + model_name
  final_path = osp.join(output_dir,f"{video_name}_{suffix}.{ext}")
  fps = fps if len(fps)!=0 else None

  if (osp.exists(final_path)):
    # return gr.Video.update(value=final_path)
    os.remove(final_path)

  args = structify({
    "input":video_path,
    "output":output_dir,
    "video_name":video_name,
    "suffix":suffix,
    "model_name":model_name,
    "denoise_strength":denoise_strength,
    "outscale":outscale,
    "tile":tile,
    "tile_pad":tile_pad,
    "pre_pad":pre_pad,
    "face_enhance":face_enhance,
    "fp32":fp32,
    "alpha_upsampler":alpha_upsampler,
    "gpu_id":gpu_id,
    "fps":fps,
    "ffmpeg_bin":ffmpeg_bin,
    "extract_frame_first":extract_frame_first,
    "num_process_per_gpu":num_process_per_gpu
  })

  print("output: " + osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4'))

  if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
    is_video = True
  else:
    is_video = False

  if is_video and args.input.endswith('.flv'):
    mp4_path = args.input.replace('.flv', '.mp4')
    os.system(f'{ffmpeg_bin} -i {args.input} -codec copy {mp4_path}')
    args.input = mp4_path

  if args.extract_frame_first and not is_video:
    args.extract_frame_first = False

  irv.run(args)

  if args.extract_frame_first:
    tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
    shutil.rmtree(tmp_frames_folder)

  return gr.Video.update(value=final_path)


with gr.Blocks(title="Real-ESRGAN") as demo:
  with gr.Row():
    with gr.Column():
      model_name = gr.Dropdown(label="Model name", value="RealESRGAN_x4plus", interactive=True, choices=list(models.keys()))
    with gr.Column(visible=False) as denoise_strength_box:
      denoise_strength = gr.Slider(0, 1, value=0.5, label="Denoise strength", visible=True, info="0 for weak denoise (keep noise), 1 for strong denoise ability.Only used for the realesr-general-x4v3 model", interactive=True),
    with gr.Column() as denoise_strength_box_pos:
      model_name.change(fn=lambda v: (gr.update(visible=(v == "realesr-general-x4v3"))), inputs=model_name, outputs=denoise_strength_box)
      model_name.change(fn=lambda v: (gr.update(visible=(v != "realesr-general-x4v3"))), inputs=model_name, outputs=denoise_strength_box_pos)
  with gr.Tabs() as tab:
    with gr.TabItem("Restore Image"):
      with gr.Row():
        with gr.Column():
          image_input = gr.Image(label="Input", image_mode="RGBA").style(height=300)
        with gr.Column():
          image_output = gr.Image(label="Output", interactive=False, image_mode="RGBA").style(height=300)
          restore_image_button = gr.Button("Restore", variant="primary")
    with gr.TabItem("Restore Video"):
      with gr.Row():
        with gr.Column():
          video_input = gr.Video(label="Input")
          with gr.Row():
            with gr.Column():
              with gr.Row():
                fps = gr.Text(label="FPS of the output video", interactive=True)
                extract_frame_first = gr.Checkbox(label="Extract frame first", info="If you encounter ffmpeg error when using multi-processing, you can turn this option on.", interactive=True)
              ffmpeg_bin = gr.Text(label="The path to ffmpeg",value="ffmpeg", interactive=True)
            with gr.Column():
              num_process_per_gpu = gr.Slider(1, 24, value=1, step=1, label="num_process_per_gpu", info="The total number of process is num_gpu * num_process_per_gpu. The bottleneck of the program lies on the IO, so the GPUs are usually not fully utilized. To alleviate this issue, you can use multi-processing by setting this parameter. As long as it does not exceed the CUDA memory", interactive=True)
        with gr.Column():
          video_output = gr.Video(label="Output", interactive=False)
          restore_video_button = gr.Button("Restore", variant="primary")
  with gr.Row():
    with gr.Column():
      outscale = gr.Slider(1, 4, value=2, step=1, label="Outscale", info="The final upsampling scale of the image", interactive=True)
      with gr.Row():
        tile = gr.Slider(0, 100, value=0, step=1, label="Tile size", info="0 for no tile during testing", interactive=True),
        tile_pad = gr.Slider(0, 100, value=10, step=1, label="Tile padding", info="Tile padding", interactive=True),
        pre_pad = gr.Slider(0, 100, value=0, step=1, label="Pre padding", info="size at each border", interactive=True),
      with gr.Row():
        face_enhance = gr.Checkbox(label="Face enhance", info="Use GFPGAN to enhance face")
        fp32 = gr.Checkbox(label="Use fp32", info="Default: fp16 (half precision).", interactive=True)
        alpha_upsampler = gr.Radio(choices=["realesrgan", "bicubic"], value="realesrgan", label="Alpha upsampler", interactive=True)
      gpu_id = gr.Text(label="GPU id", info="gpu device to use (default=None) can be 0,1,2 for multi-gpu")
    with gr.Column():
      None
  public_inputs = [model_name, denoise_strength[0], outscale, tile[0], tile_pad[0], pre_pad[0], face_enhance, fp32, alpha_upsampler, gpu_id]
  restore_image_button.click(fn=restore_image, outputs=image_output, inputs=[image_input, *public_inputs], api_name="restore_image")
  restore_video_button.click(fn=restore_video, outputs=video_output, inputs=[video_input, *public_inputs, fps, ffmpeg_bin, extract_frame_first, num_process_per_gpu], api_name="restore_video")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
  parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
  parser.add_argument("--server-name", type=str,   help="Sets hostname of server", default=None)
  parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
  parser.add_argument("--tls-keyfile", type=str, help="Partially enables TLS, requires --tls-certfile to fully function", default=None)
  parser.add_argument("--tls-certfile", type=str, help="Partially enables TLS, requires --tls-keyfile to fully function", default=None)
  parser.add_argument("--gradio-debug",  action='store_true', help="launch gradio with --debug option")
  parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
  parser.add_argument("--gradio-auth-path", type=str, help='set gradio authentication file path ex. "/path/to/auth/file" same auth format as --gradio-auth', default=None)
  parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
  args = parser.parse_args()

  if args.server_name:
    server_name = args.server_name
  else:
    server_name = "0.0.0.0" if args.listen else None

  gradio_auth_creds = []
  if args.gradio_auth:
    gradio_auth_creds += [x.strip() for x in args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip()]
  if args.gradio_auth_path:
    with open(args.gradio_auth_path, 'r', encoding="utf8") as file:
      for line in file.readlines():
        gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]

  demo.launch(
    share=args.share,
    server_name=args.server_name,
    server_port=args.port,
    ssl_keyfile=args.tls_keyfile,
    ssl_certfile=args.tls_certfile,
    debug=args.gradio_debug,
    auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
    inbrowser=args.autolaunch,
  )
