class Node:
  def __init__(self, attribute=None, label=None):
    self.attribute = attribute
    self.label = label
    self.children = {}
  
  def setAttribute(attribute):
    self.attribute = attribute

  def setLabel(label):
    self.label = label
  
  def addChildren(attributeValue, node):
    self.children[attributeValue] = node