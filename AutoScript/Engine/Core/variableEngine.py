from typing import reveal_type


class VariableManager:
    def __init__(self):
        self.variables = {}
    def addVariable(self,name,type,value):
        self.variables[name] = Variable(type,value)
    def get(self,name):
        return f"{name}: {self.variables[name].get()}"
    def setVariable(self,name,value,type):
        self.variables[name] = Variable(type,value)



class Variable:
    def __init__(self,type = "NoneType",value = "None"):
        self.type = type
        self.value = value
    def get(self):
        return f"({self.type}) {self.value}"