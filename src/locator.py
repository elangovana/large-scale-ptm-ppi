import importlib


class Locator:
    """
Loads classes dynamically if the have a constrcutore with zero args
    """

    def get(self, module_class_name):
        module_parts = module_class_name.split(".")
        module_name = ".".join(module_parts[0:-1])
        class_name = module_parts[-1]

        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        instance = class_()

        return instance