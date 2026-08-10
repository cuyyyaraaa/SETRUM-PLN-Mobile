"""
train_indobert.py — Fine-tuning IndoBERT untuk Layer 1/2/3 dan Sentimen
=======================================================================
Jalankan di Google Colab (GPU). Hasil model di-download lalu ditaruh di:
    models/layer1/  models/layer2/  models/layer3/  models/sentiment/

Menangani DATA TIMPANG dengan class weighting (penting untuk Layer 3).

Cara pakai (Colab):
    !pip install transformers datasets scikit-learn pandas torch
    # upload data berlabel (CSV: kolom 'text','label')
    python train_indobert.py --data data_layer3.csv --out models/layer3 --epochs 5

Kolom CSV wajib: 'text' (ulasan, sebaiknya sudah dinormalisasi), 'label' (kategori).
"""
import argparse, os, numpy as np, pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import Dataset

BASE_MODEL = "indobenchmark/indobert-base-p1"  # IndoBERT (Koto/IndoNLU). Bisa diganti.


class WeightedTrainer(Trainer):
    """Trainer dgn class weighting untuk meredam ketimpangan kelas."""
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV dgn kolom text,label')
    ap.add_argument('--out', required=True, help='folder output model')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--maxlen', type=int, default=128)
    ap.add_argument('--no_weight', action='store_true', help='matikan class weighting')
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    labels = sorted(df['label'].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    df['y'] = df['label'].map(l2i)
    print(f"Data: {len(df)} baris, {len(labels)} kelas")
    print(df['label'].value_counts())

    # split STRATIFIED (wajib untuk data timpang)
    tr, te = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=42)
    print(f"Train {len(tr)} | Test {len(te)}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def enc(batch):
        return tok(batch['text'], truncation=True, max_length=args.maxlen, padding='max_length')

    ds_tr = Dataset.from_pandas(tr[['text', 'y']].rename(columns={'y': 'labels'})).map(enc, batched=True)
    ds_te = Dataset.from_pandas(te[['text', 'y']].rename(columns={'y': 'labels'})).map(enc, batched=True)
    cols = ['input_ids', 'attention_mask', 'labels']
    ds_tr.set_format('torch', columns=cols); ds_te.set_format('torch', columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), id2label=i2l, label2id=l2i)

    cw = None
    if not args.no_weight:
        w = compute_class_weight('balanced', classes=np.arange(len(labels)), y=tr['y'].values)
        cw = torch.tensor(w, dtype=torch.float)
        print("Class weights:", dict(zip(labels, w.round(2))))

    def metrics(p):
        pred = np.argmax(p.predictions, axis=1)
        return {'accuracy': accuracy_score(p.label_ids, pred),
                'macro_f1': f1_score(p.label_ids, pred, average='macro', zero_division=0),
                'weighted_f1': f1_score(p.label_ids, pred, average='weighted', zero_division=0),
                'kappa': cohen_kappa_score(p.label_ids, pred)}

    targs = TrainingArguments(
        output_dir=args.out + '_ckpt', num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        learning_rate=args.lr, eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='macro_f1',
        logging_steps=20, report_to='none')

    trainer = WeightedTrainer(class_weights=cw, model=model, args=targs,
                              train_dataset=ds_tr, eval_dataset=ds_te,
                              compute_metrics=metrics)
    trainer.train()

    print("\n=== EVALUASI AKHIR ===")
    ev = trainer.evaluate()
    for k in ['eval_accuracy', 'eval_macro_f1', 'eval_weighted_f1', 'eval_kappa']:
        print(f"  {k:18s}: {ev[k]:.4f}")

    pred = np.argmax(trainer.predict(ds_te).predictions, axis=1)
    print("\n", classification_report(te['y'].values, pred, target_names=labels, zero_division=0))

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"\nModel disimpan ke {args.out}/")


if __name__ == '__main__':
    main()
