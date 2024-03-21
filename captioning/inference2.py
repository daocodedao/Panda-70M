from video_llama.conversation.conversation_video import Chat as chat
from video_llama.conversation.conversation_video import default_conversation


# if args.model_type == 'vicuna':
chat_state = default_conversation.copy()
# else:
#     chat_state = conv_llava_llama_2.copy()

video_path = "/data/work/Panda-70M/splitting/outputs/video1.0.mp4"
chat_state.system = ""
img_list = None
llm_message = chat.upload_video(video_path , chat_state, img_list)

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