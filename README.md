# DPEfficR

*DPEfficR*, a data and parameter efficient method for building API recommendation systems. 



## Design Overview
<div align="center">    
 <img src="https://github.com/Prompt-Hijacking/API-Recommendation/blob/main/overview.png?raw=true" width="780" height="360" alt="Design Overview"/><br/>
</div> 

*DPEfficR* comprises three main modules:(1) a data selection module for curating diverse unlabeled training data, (2) a task-specific parameter tuning module that fine-tunes a pre-trained code model with fewer parameters using manually labeled data, and (3) a runtime API selection module to ensure the integrity of recommended API sequences.

## Data

Please kindly find the data [link](https://smu-my.sharepoint.com/personal/tingzhang_2019_phdcs_smu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ftingzhang%5F2019%5Fphdcs%5Fsmu%5Fedu%5Fsg%2FDocuments%2FSANER%2D23%2Ddata%2Dv1&ga=1).


## File Structure
* **src** -main source codes.
  * **./src/run_XXX.py** - the implementation of DPEfficR.
  * **./src/test/test_XXX.py** -the testing of DPEfficR.
  * **./src/test/runtime_selection.py** -a method to filter the most correct API
  * **./src/test/Bleu.py** -calculate the Bleu score
  * **./src/test/Cider.py** -calculate the Cider score
  * **./src/test/Rouge.py** -calculate the Rouge score
* **data_processing.py** -data processing procedure

## Setup

We recommend to use ``conda`` to manage the python libraries and the requirement libraries are listed in ``requirements.txt``. So run ``pip install -r requirements.txt``.

## Train

Example:

```python
CUDA_VISIBLE_DEVICES=1 python run_annotation.py --peft_config_type XXX --output_dir '19-Oct-annotation'
```

## Test

Example:

```python
CUDA_VISIBLE_DEVICES=2 python test_annotation.py
```

