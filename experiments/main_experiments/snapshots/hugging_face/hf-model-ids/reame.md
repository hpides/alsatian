The snapshots are listed in this format: miriaiml/detr-resnet-50_finetuned_cppe5

- username: miriaiml
- model: detr-resnet-50_finetuned_cppe5

- during crawling we replaced all "/" with "-" and later changed the first occurance of "-" back to "/"
- this might lead to errors when the username contains a "-"
- e.g. 
- miriaiml/2-detr-resnet-50_finetuned_cppe5 might not existsi because the actual username is miriaiml-2
- in this scenario replace the model id as follows
- "miriaiml/2-detr-resnet-50_finetuned_cppe5" -> "miriaiml-2/detr-resnet-50_finetuned_cppe5"