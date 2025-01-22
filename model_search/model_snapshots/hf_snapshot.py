

class HFModelSnapshot:
    def __init__(self, base_model_id:str, model_id:str, hf_caching_path:str):
        self.hf_caching_path = hf_caching_path
        self.base_model_id = base_model_id
        self.model_id = model_id
        self.id = model_id.replace("/","-")

