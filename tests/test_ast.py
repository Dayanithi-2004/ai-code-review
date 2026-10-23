import ast

source = """
class MyClass:
    def method_a(self):
        pass

    def method_b(self):
        print("changed")
"""
tree = ast.parse(source)
for node in ast.walk(tree):
    if hasattr(node, 'lineno'):
        print(f"{type(node).__name__}: {node.lineno}-{node.end_lineno}")
