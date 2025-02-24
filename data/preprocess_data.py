from modeling import build_tokenizer
import random

table = {ord(f): ord(t) for f, t in zip(
    u'，。！？：【】（）％＃＠＆１２３４５６７８９０',
    u',.!?:[]()%#@&1234567890')}


def punctuation_format(text: str):
    # Replace non-breaking space with space
    # text = text.strip() + '\n'
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # change chinese punctuation to english ones
    # text = text.translate(table)
    if not text.endswith("\n"):
        text += "\n"
    return text


def format_eol(text):
    if not text.endswith("\n"):
        text += "\n"
    return text


def get_white_space():
    r = random.random()
    return '' if r < 0.33 else (' ' if r < 0.66 else '\n')


def gen_prompt(tokenizer, data, graph_pad_id, graph_token_num):
    # randomly select [graph tokens, question] or [question, graph tokens]
    if random.random() < 0.5:
        return tokenizer.encode(f"{data['question']}{get_white_space()}", add_special_tokens=False) + [graph_pad_id] * graph_token_num + tokenizer.encode('\n', add_special_tokens=False)
    else:
        return [graph_pad_id] * graph_token_num + tokenizer.encode(f"{get_white_space()}{format_eol(data['question'])}", add_special_tokens=False)


class Encoder(object):
    def __init__(self, args):
        self.args = args
        # seq_length - 1 for shifting
        self.seq_length = args.seq_length - 1

    def initializer(self):
        self.tokenizer = build_tokenizer(self.args)
        
        self.HUMAN = 'human'
        self.BOT = 'bot'
        self.SYSTEM = 'system'
        self.ROLE_START_MARKER = '<s>'
        self.ROLE_END_MARKER = '\n'

        self.human_marker_ids = self.tokenizer.encode(f"{self.ROLE_START_MARKER}{self.HUMAN}{self.ROLE_END_MARKER}", add_special_tokens=False)
        self.bot_marker_ids = self.tokenizer.encode(f"{self.ROLE_START_MARKER}{self.BOT}{self.ROLE_END_MARKER}", add_special_tokens=False)
        self.system_marker_ids = self.tokenizer.encode(f"{self.ROLE_START_MARKER}{self.SYSTEM}{self.ROLE_END_MARKER}", add_special_tokens=False)
        self.sft_end_marker_ids = [self.tokenizer.eos_token_id]
        self.role_to_markerid = {self.HUMAN: self.human_marker_ids, self.BOT: self.bot_marker_ids, self.SYSTEM: self.system_marker_ids}

        self.default_system_ids = self.system_marker_ids + self.tokenizer.encode('You are an AI code assistant. You will be given a task. You must provide an accurate answer according to the requirements.\n', add_special_tokens=False)
    
    def padding(self, input_ids, loss_mask):
        pad_id = self.tokenizer.pad_token_id
        assert len(input_ids) <= self.seq_length, f"padding sequence: {len(input_ids)} > {self.seq_length}"
        input_ids += [pad_id] * (self.seq_length - len(input_ids))
        loss_mask += [0] * (self.seq_length - len(loss_mask))
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask
        }


class UniformEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)

    def encode_graph(self, data):
        input_ids, loss_mask = [], []

        if self.args.mode == 'ft':
            # system
            # input_ids += self.default_system_ids
            # loss_mask += [0] * len(self.default_system_ids)
            # human
            content_ids = gen_prompt(self.tokenizer, data, self.args.graph_pad_id, self.args.graph_token_num)
            input_ids += self.human_marker_ids + content_ids
            loss_mask += [0] * (len(self.human_marker_ids) + len(content_ids))
            # bot
            content_ids = self.tokenizer.encode(data['bot'], add_special_tokens=False) + self.sft_end_marker_ids
            input_ids += self.bot_marker_ids + content_ids
            loss_mask += [0] * len(self.bot_marker_ids) + [1] * len(content_ids)

        elif self.args.mode == 'pt':
            # graph
            content_ids = [self.args.graph_pad_id] * self.args.graph_token_num
            input_ids += content_ids
            loss_mask += [0] * len(content_ids)
            # source code
            content_ids = self.tokenizer.encode(data['source'], add_special_tokens=False) + self.sft_end_marker_ids
            input_ids += content_ids
            loss_mask += [1] * len(content_ids)
            
        else:
            raise NotImplementedError()

        assert len(input_ids) == len(loss_mask)
        if len(input_ids) <= self.seq_length:
            features = self.padding(input_ids, loss_mask)
            features['node_ids'] = data['node_ids']
            features['edge_index'] = data['edge_index']
            return features

        # drop if too long
        else:
            return None
        
    def encode_text(self, data):
        input_ids, loss_mask = [], []

        # system
        # input_ids += self.default_system_ids
        # loss_mask += [0] * len(self.default_system_ids)
        # human
        content_ids = self.tokenizer.encode(data['human'], add_special_tokens=False)
        input_ids += self.human_marker_ids + content_ids
        loss_mask += [0] * (len(self.human_marker_ids) + len(content_ids))
        # bot
        content_ids = self.tokenizer.encode(data['bot'], add_special_tokens=False) + self.sft_end_marker_ids
        input_ids += self.bot_marker_ids + content_ids
        loss_mask += [0] * len(self.bot_marker_ids) + [1] * len(content_ids)

        assert len(input_ids) == len(loss_mask)
        if len(input_ids) <= self.seq_length:
            features = self.padding(input_ids, loss_mask)
            return features

        # drop if too long
        else:
            return None