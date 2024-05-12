# DPEfficR

*DPEfficR*, a data and parameter efficient method for building API recommendation systems. 



## Design Overview
<div align="center">    
 <img src="https://github.com/Prompt-Hijacking/API-Recommendation/blob/main/fig/overview.png?raw=true" width="880" height="360" alt="Design Overview"/><br/>
</div> 

*DPEfficR* comprises three main modules:(1) a data selection module for curating diverse unlabeled training data, (2) a prompt tuning module that fine-tunes a pre-trained code model with fewer parameters using manually labeled data, and (3) a runtime API selection module to ensure the integrity of recommended API sequences.

## Data

Please kindly find the data [link](https://smu-my.sharepoint.com/:u:/g/personal/ivanairsan_smu_edu_sg/EVwqxPdn_0tDl-Zhse0KCj4Bty3iEGuoAWWy2inAlhYTnQ?e=Htfks2).


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
CUDA_VISIBLE_DEVICES=1 python run_annotation.py --peft_config_type IA3 --output_dir '19-Oct-annotation'
```

## Test

Example:

```python
CUDA_VISIBLE_DEVICES=2 python test_annotation.py
```

