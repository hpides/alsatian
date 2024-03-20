- want to find out if it makes a difference using our model merge functionality vs using a model that is already merged
  as given in pytorch code
- output:

```
num params: 41283328
num params: 41283328
nat merged: [6.812114746093751, 4.9401489257812505, 4.9695078125, 5.01284228515625, 5.0364384765625, 5.08271728515625, 5.12260302734375, 5.1480258789062505, 5.16630419921875, 5.19027001953125]
meth merged: [4.92790185546875, 5.051697265625, 4.989001953125, 5.0114345703125, 5.1698852539062505, 5.227638671875, 5.2412705078125, 5.1549267578125, 5.16868212890625, 5.1940146484375]
```

- both models have the same number of parameters and converge to very similar inference times
- thus we consider them equivalent
