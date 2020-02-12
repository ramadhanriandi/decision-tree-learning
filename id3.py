import math

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


def entropy(data, target_attribute, filter_value_attribute=None):
  parsed_data = data

  if filter_value_attribute is not None: 
    #get parsed_data here based on filter_value_attribute only

  parsed_value_target = {}
  total_value_target = 0
  
  for i in parsed_data[target_attribute]:
    if i is not None:
      if i not in parsed_value_target:
        parsed_value_target[i] = 1
      else:
        parsed_value_target[i] += 1

      total_value_target += 1
  
  log_result = 0

  for i in parsed_value_target:
    log_result += float(parsed_value_target[i])/total_value_target * math.log((float(parsed_value_target[i])/total_value_target), 2)
  
  return -1 * log_result

# hasn't handle after universal entropy
def information_gain(data, previous_entropy_result, previous_attribute, target_attribute):
  gain_result = 0
  attribute_entropy_result = 0
  parsed_attribute_count = {}
  total_attribute_count = 0
  
  for i in data[attribute]:
    if i is not None:
      if i not in parsed_attribute_count:
        parsed_attribute_count[i] = 1
      else:
        parsed_attribute_count[i] += 1

      total_value_target += 1

  for i in parsed_attribute_count:
    attribute_entropy_result += float(parsed_attribute_count[i])/total_attribute_count * entropy(data, target_attribute, previous_attribute)    

  gain_result += previous_entropy_result + (-1 * attribute_entropy_result)

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