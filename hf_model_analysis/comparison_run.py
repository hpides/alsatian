import os
from datetime import datetime

from global_utils.file_names import clear_directory
from hf_model_analysis.hugging_face_crawler import HFCrawler
from hf_model_analysis.model_comparator import ModelComparator


class ModelComparisonRun:

    def __init__(self, base_model_id, base_model_class, base_model_url, start_page_num, end_page_num,
                 result_output_path, hf_cache_path):
        self.base_model_id = base_model_id
        self.base_model_class = base_model_class
        self.base_model_url = base_model_url
        self.start_page_num = start_page_num
        self.end_page_num = end_page_num
        self.result_output_path = result_output_path
        self.hf_crawler = HFCrawler(self.base_model_url, self.start_page_num, self.end_page_num)
        self.fine_tuned_cache_dir = os.path.join(hf_cache_path, base_model_id)
        clear_directory(self.fine_tuned_cache_dir, delete_subdirectories=True)
        self.base_model = self.base_model_class.from_pretrained(base_model_id)

    def start_comparison(self):
        model_ids, model_metadata = self.hf_crawler.get_model_info()

        for fine_tuned_model_id in model_ids:
            try:
                fine_tuned_model = self.base_model_class.from_pretrained(
                    fine_tuned_model_id, cache_dir=self.fine_tuned_cache_dir)
                mc = ModelComparator(fine_tuned_model, f"base_model_id: {self.base_model_id}",
                                     self.base_model, f"derived_model_id: {fine_tuned_model_id}")
                report = mc.generate_comparison_report()

                current_timestamp = datetime.now()
                timestamp = current_timestamp.strftime("%Y-%m-%d-%H:%M:%S")
                file_name = f"{timestamp};compare;{self.base_model_id.replace('/', '-')};{fine_tuned_model_id.replace('/', '-')}.txt"
                file_path = os.path.join(self.result_output_path, file_name)

                print(f"WRITE: {file_path}")
                with open(file_path, 'w') as file:
                    file.write(report)
                clear_directory(self.fine_tuned_cache_dir, delete_subdirectories=True)
            except:
                clear_directory(self.fine_tuned_cache_dir, delete_subdirectories=True)
