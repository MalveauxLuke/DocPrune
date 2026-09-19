import torch

from docprune.stage2.qwen import prepare_prompt


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        assert not add_special_tokens and return_offsets_mapping
        return dict(input_ids=[ord(c) for c in text],
            offset_mapping=[(i,i+1) for i in range(len(text))])
    def decode(self,ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return ''.join(chr(i) for i in ids)


class Processor:
    def __init__(self):
        self.tokenizer=Tokenizer();self.message=None
    def apply_chat_template(self,message,tokenize=False,add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        self.message=message
        return ''.join(part['text'] for turn in message for part in turn['content'] if part['type']=='text')
    def __call__(self,*,text,images,return_tensors,padding):
        assert return_tensors=='pt' and padding is False and len(text)==1
        return dict(input_ids=torch.tensor([[ord(c) for c in text[0]]]),
            image_grid_thw=torch.tensor([[1,4,4]]*len(images)),
            pixel_values=torch.zeros(16*len(images),3))


def test_default_prompt_remains_raw_question_then_images():
    p=Processor();result=prepare_prompt(p,'Where?',[object()])
    assert [x['type'] for x in p.message[0]['content']]==['text','image']
    assert p.message[0]['content'][0]['text']=='Where?'
    assert result.question_positions.tolist()==list(range(6))


def test_task_prompt_uses_official_roles_but_tracks_only_query_tokens():
    p=Processor();result=prepare_prompt(p,'Where?',[object(),object()],
        system='System frame.',instruction='Find answer evidence.')
    assert [turn['role'] for turn in p.message]==['system','user']
    texts=[x['text'] for x in p.message[1]['content'] if x['type']=='text']
    assert texts==['<Instruct>: Find answer evidence.','<Query>:\n','Where?','\n<Document>:']
    ids=result.input_ids[0].tolist();positions=result.question_positions.tolist()
    assert [ids[i] for i in positions]==[ord(c) for c in 'Where?']
    assert len(positions)==len('Where?')


def test_task_prompt_does_not_require_standalone_query_tokens_to_match_context():
    class BoundarySensitiveTokenizer(Tokenizer):
        def encode(self,text,add_special_tokens=False):
            # Simulate a tokenizer whose first standalone token changes when
            # the same text is encoded inside the rendered prompt.
            result=super().encode(text,add_special_tokens=add_special_tokens)
            return [result[0]+1000,*result[1:]]

    p=Processor();p.tokenizer=BoundarySensitiveTokenizer()
    result=prepare_prompt(p,'Where?',[object()],system='System frame.',
        instruction='Find answer evidence.')
    ids=result.input_ids[0].tolist();positions=result.question_positions.tolist()
    assert ''.join(chr(ids[i]) for i in positions)=='Where?'
