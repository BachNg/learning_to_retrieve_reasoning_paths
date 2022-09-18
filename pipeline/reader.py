import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import XLMRobertaForQuestionAnswering, XLMRobertaTokenizer
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.data.processors.squad import squad_convert_examples_to_features

from reader.modeling_reader import BertForQuestionAnsweringConfidence
from reader.rc_utils import read_squad_style_hotpot_examples, \
    convert_examples_to_features, write_predictions_yes_no_beam

import collections

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits"])

class Reader:
    def __init__(self,
                 args,
                 device):

        print('initializing Reader...', flush=True)
        # self.model = BertForQuestionAnsweringConfidence.from_pretrained(args.reader_path,  num_labels=4, no_masking=True)
        # self.tokenizer = BertTokenizer.from_pretrained(args.reader_path, args.do_lower_case)
        self.model = XLMRobertaForQuestionAnswering.from_pretrained(args.reader_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        self.device = device
        
        self.model.to(device)
        self.model.eval()
        print('Done!', flush=True)

    def convert_retriever_output(self,
                                 retriever_output):

        selected_paras_top_n = {str(item["q_id"]): item["topk_titles"]
                                for item in retriever_output}

        context_dic = {str(item["q_id"]): item["context"]
                       for item in retriever_output}

        squad_style_data = {'data': [], 'version': '1.1'}

        retrieved_para_dict = {}

        for data in retriever_output:
            example_id = data['q_id']
            question_text = data['question']
            pred_para_titles = selected_paras_top_n[example_id]

            for selected_paras in pred_para_titles:
                title, context = "", ""

                for para_title in selected_paras:
                    paragraphs = context_dic[example_id][para_title]
                    context += paragraphs

                    title = para_title
                    context += " "
                # post process to remove unnecessary spaces.
                if context[0] == " ":
                    context = context[1:]
                if context[-1] == " ":
                    context = context[: -1]

                context = context.replace("  ", " ")

                squad_example = {'context': context, 'para_titles': selected_paras,
                                 'qas': [{'question': question_text, 'id': example_id}]}
                squad_style_data["data"].append(
                    {'title': title, 'paragraphs': [squad_example]})

        return squad_style_data
    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    def predict_new(self,
                retriever_output,
                args):
        squad_style_data = self.convert_retriever_output(retriever_output)
        

        e = read_squad_style_hotpot_examples(squad_style_hotpot_dev=squad_style_data,
                                             is_training=False,
                                             version_2_with_negative=False,
                                             store_path_prob=False)
        dev_features, dev_dataset = squad_convert_examples_to_features(e, 
                                                       self.tokenizer, 
                                                       max_seq_length = 378, 
                                                       doc_stride = 128,
                                                       max_query_length = 64,
                                                       is_training = False,
                                                       return_dataset = 'pt',
                                                       threads = 10
                                                       )
        eval_sampler = SequentialSampler(dev_dataset)
        eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=12)
        all_results = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                del inputs["token_type_ids"]
                example_indices = batch[3]
                outputs = self.model(**inputs)
            # print('VVVVVVVVV', outputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = dev_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
    #             for output in outputs:
    #                 print(output)
                output = [self.to_list(output[i]) for output in outputs.to_tuple()]
    #             output = [to_list(output) for output in outputs]
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )
                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)
        print(all_results)

    def predict(self,
                retriever_output,
                args):

        squad_style_data = self.convert_retriever_output(retriever_output)

        # print("CCCCCCCCCCCCCCCCSSSSSSSSSSSSSSS",squad_style_data)

        e = read_squad_style_hotpot_examples(squad_style_hotpot_dev=squad_style_data,
                                             is_training=False,
                                             version_2_with_negative=False,
                                             store_path_prob=False)
        # print("KKKKKKKKKKKKKKFFFFFFFFFFFFFFFFFF",e)
        features = convert_examples_to_features(
            examples=e,
            tokenizer=self.tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            quiet = True)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        all_results = []

        f_offset = 0
        for input_ids, input_masks, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_masks = input_masks.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_switch_logits = self.model(input_ids, segment_ids, input_masks)

            for i in range(input_ids.size(0)):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                switch_logits = batch_switch_logits[i].detach().cpu().tolist()
                # print('aaaaaaaaaaaaaaa', start_logits)
                # print('bbbbbbbbbbb', end_logits)
                # print('sssssssss', switch_logits)
                eval_feature = features[f_offset+i]
                unique_id = int(features[f_offset+i].unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits,
                                             switch_logits=switch_logits))
            f_offset += input_ids.size(0)
            
        return write_predictions_yes_no_beam(e, features, all_results,
                                             args.n_best_size, args.max_answer_length,
                                             args.do_lower_case, None,
                                             None, None, False,
                                             False, None,
                                             output_selected_paras=True,
                                             quiet = True)
