import os
import json
import torch
import torch.nn.functional as F
from seqeval.metrics import f1_score
from pytorch_pretrained_bert import BertForTokenClassification

class BertNER(BertForTokenClassification):
    """Use BertForTokenClassification for NER"""
    def __init__(self, config, tag_to_ix):
        super(BertNER, self).__init__(config, len(tag_to_ix))
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {self.tag_to_ix[tag] : tag for tag in self.tag_to_ix}

    def predict(self, input_ids, input_mask):

        with torch.no_grad():
          logits = self.forward(input_ids, None, input_mask)

        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        return logits.detach()

    def f1_eval(self, dataloader, device):

        all_tag_seqs = []
        all_tag_seqs_pred = []
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            logits= self.predict(input_ids, input_mask)
            for i,mask in enumerate(input_mask):
              temp_1 =  []
              temp_2 = []
              for j, m in enumerate(mask):
                  if m:
                      if self.ix_to_tag[label_ids[i][j].item()] != "X":
                          temp_1.append(self.ix_to_tag[label_ids[i][j].item()])
                          temp_2.append(self.ix_to_tag[logits[i][j].item()])
                  else:
                      break

            all_tag_seqs.append(temp_1)
            all_tag_seqs_pred.append(temp_2)

        f1 = f1_score(all_tag_seqs, all_tag_seqs_pred)
          
        return f1

    def save(self, output_dir):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # save model weights
        output_model_file = os.path.join(output_dir, 'model.pt')
        torch.save(self.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, 'config.json')
        with open(output_config_file, 'w') as f:
            f.write(self.config.to_json_string())

        meta_config = {"num_labels":len(self.tag_to_ix), "label_map":self.ix_to_tag}
        json.dump(meta_config,open(os.path.join(output_dir,"meta_config.json"),"w"), indent=3)
