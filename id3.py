class Node:
  def __init__(self, attribute=None, label=None):
    self.attribute = attribute
    self.label = label
    self.children = {}
  
  def setAttribute(self, attribute):
    self.attribute = attribute

  def setLabel(self, label):
    self.label = label
  
  def addChildren(self, attributeValue, node):
    self.children[attributeValue] = node


# Try to build a tree
n1 = Node('Outlook')
n2 = Node('Humidity')
n3 = Node(label = 'Yes')
n4 = Node('Wind')
n5 = Node(label = 'Yes')
n6 = Node(label = 'No')

n1.addChildren('Sunny', n2)
n1.addChildren('Overcast', n3)
n1.addChildren('Rain', n4)
n2.addChildren('High', n5)
n2.addChildren('Normal', n6)

print(n1.children['Sunny'].children['High'].label)