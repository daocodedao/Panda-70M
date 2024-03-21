from video_llama.conversation.conversation_video import Chat
from video_llama.conversation.conversation_video import default_conversation
from video_llama.common.registry import registry
from video_llama.common.config import Config
import argparse

# if args.model_type == 'vicuna':
chat_state = default_conversation.copy()
# else:
#     chat_state = conv_llava_llama_2.copy()

video_path = "/data/work/Panda-70M/splitting/outputs/video1.0.mp4"
chat_state.system = ""
img_list = []

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--cfg-path", default="eval_configs/panda70M_eval.yaml", help="path to configuration file.")
parser.add_argument("--output-json", default=None, help="output json file. Leave none to print out the results.")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

parser.add_argument("--prompt-list", default=None, help="list of correponding input prompts. Leave none if no prompt input.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
args = parser.parse_args()
cfg = Config(args)


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
print(f"model_cls:{model_cls}")
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
print(f"model.eval()")
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

# chat = Chat("lmsys/vicuna-7b-v1.5", device='cuda:0')
llm_message = chat.upload_video(video_path=video_path, conv=chat_state, img_list=img_list)

while True:
    user_message = input("User/ ")

    chat.ask(user_message, chat_state)

    num_beams = 2
    temperature = 1.0

    llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
    print(chat_state.get_prompt())
    print(chat_state)
    print(llm_message)