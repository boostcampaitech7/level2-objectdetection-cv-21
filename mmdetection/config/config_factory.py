# config_factory.py
import importlib

def create_config(model_name, max_epochs=25):
    module_name = f"config.{model_name}_config"
    class_name = f"{model_name}_config"
    module = importlib.import_module(module_name)
    config_class = getattr(module, class_name)
    config = config_class(max_epochs=max_epochs)
    return config.build_config(), config.model_name, config.output_dir

