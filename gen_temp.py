import string
import torch
import pandas as pd
from tqdm import tqdm
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = T5ForConditionalGeneration.from_pretrained("Title-Gen")
tokenizer = T5Tokenizer.from_pretrained("Title-Gen")
model.to(DEVICE)


def get_title(prefix, input_text):

    input_ids = tokenizer(prefix+": "+input_text ,return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    summary_text_ids = model.generate(
        input_ids=input_ids["input_ids"].to(DEVICE),
        attention_mask=input_ids["attention_mask"].to(DEVICE),
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.2,
        top_k=5,
        top_p=0.95,
        max_length=48,
        min_length=2,
        num_beams=3,
    )
    title = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    if(title[-1] in string.punctuation):
        title = title[:-1] + " " +title[-1]
    return title

if __name__ == '__main__':
    prefix = "Python"
    body = """
    I have a Custom User model that takes user ip address. I want to add the IP address of the user upon completion of the sign up form. Where do I implement the below code? I am not sure whether to put this into my forms.py or views.py file.
    I expect to be able to save the user's ip address into my custom user table upon sign up.
    """

    code = """
    def get_client_ip(request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    """
    input_text = ' '.join(body.split()[:256]) + " <code> " + ' '.join(code.split()[:256])
    title = get_title(prefix, input_text)
    print(title)
    # df = pd.read_csv("test.csv")
    # prefix_list = df['prefix'].tolist()
    # input_text_list = df['input_text'].tolist()
    # target_text_list = df['target_text'].tolist()
    # data_dir_list = ['C#', 'Java', 'JS', 'Python']
    #
    # for data_dir in data_dir_list:
    #     comment_list = []
    #     true_list = []
    #     for i in tqdm(range(len(prefix_list))):
    #         if(prefix_list[i] == data_dir):
    #             comment = get_comment(prefix_list[i], input_text_list[i])
    #             comment_list.append(comment)
    #             true_list.append(target_text_list[i])
    #     df = pd.DataFrame(comment_list)
    #     df.to_csv(data_dir+"_pred.csv", index=False, header=None)
    #     df = pd.DataFrame(true_list)
    #     df.to_csv(data_dir+"_true.csv", index=False, header=None)
    #     from nlgeval import compute_metrics
    #
    #     metrics_dict = compute_metrics(hypothesis=data_dir+"_pred.csv",
    #                                    references=[data_dir+"_true.csv"], no_skipthoughts=True, no_glove=True)