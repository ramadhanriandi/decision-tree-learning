# import needed library
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math

#load iris dataset
data_iris = load_iris()
iris_X, iris_y = load_iris(return_X_y=True)
feature_iris = data_iris['feature_names']

#load play tennis dataset
play_tennis =  pd.read_csv('play_tennis.csv')
play_tennis = play_tennis.drop('day',axis=1)

#transform iris into dataframe
iris_X=pd.DataFrame(iris_X)
iris_y=pd.DataFrame(iris_y)

#create index so be merge
iris_X=iris_X.reset_index()
iris_y=iris_y.reset_index()
iris_y.rename(columns = {0:4}, inplace = True) 

#merge dataset iris
iris=iris_X.merge(iris_y)

#drop index
iris.drop("index",axis=1,inplace=True)
iris.rename(columns = {0:feature_iris[0],1:feature_iris[1],2:feature_iris[2],3:feature_iris[3],4:"target"}, inplace = True)

def entropy(parsed_data, target_attribute):
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
def information_gain(data, gain_attribute, target_attribute):
    gain_result = 0
    attribute_entropy_result = 0
    parsed_attribute_count = {}
    total_attribute_count = 0
    
    for i in data[gain_attribute]:
        if i is not None:
            if i not in parsed_attribute_count:
                parsed_attribute_count[i] = 1
            else:
                parsed_attribute_count[i] += 1
            
            total_attribute_count += 1
    
    for i in parsed_attribute_count:
        parsed_data = data.loc[data[gain_attribute]==i]
        attribute_entropy_result += float(parsed_attribute_count[i])/total_attribute_count * entropy(parsed_data, target_attribute)    

    gain_result += entropy(data,target_attribute) + (-1 * attribute_entropy_result)
    return gain_result

def split_information(data, gain_attribute):
    res = entropy(data, gain_attribute)
    if(res==0):
        res=0.00000001
    return res

def gain_ratio(data, gain_attribute, target_attribute):
    res = information_gain(data, gain_attribute, target_attribute) / split_information(data, gain_attribute)
    return res

def gain_ratio_continous(data, gain_attribute, target_attribute):
    res = information_gain_continous(data, gain_attribute, target_attribute)[0] / split_information(data, gain_attribute)
    return res,information_gain_continous(data, gain_attribute, target_attribute)[1]

def best_attribute(data,target_attribute,is_IG):
    gain_attribute = {
        'value': 0,
        'name': ''
    }
    
    
    for i in data.columns:
        if (i != target_attribute):
            if is_IG:
                if information_gain(data, i, target_attribute) > gain_attribute['value']:
                    gain_attribute['value'] = information_gain(data, i, target_attribute)
                    gain_attribute['name'] = i
            else:
                if gain_ratio(data, i, target_attribute) > gain_attribute['value']:
                    gain_attribute['value'] = gain_ratio(data, i, target_attribute)
                    gain_attribute['name'] = i
                

    return gain_attribute['name']

class Node:
    def __init__(self, attribute=None, label=None, vertex=None):
        self.attribute = attribute
        self.label = label
        self.vertex = vertex
        self.children = {}
        self.most_common_label = None
    
    def set_most_common_label(self, most_common_label):
        self.most_common_label = most_common_label
        
    def get_most_common_label(self):
        return self.most_common_label
        
    def setAttribute(self, attribute):
        self.attribute = attribute

    def setLabel(self, label):
        self.label = label
        
    def setVertex(self, vertex):
        self.vertex = vertex
  
    def addChildren(self, attributeValue, node):
        self.children[attributeValue] = node
    
    def getChildren(self):
        return self.children
    
    def getLabel(self):
        return self.label
    
    def getVertex(self):
        return self.vertex

def get_most_common_label(data, target_attribute):
    parsed_value_target = {}
  
    for i in data[target_attribute]:
        if i is not None:
            if i not in parsed_value_target:
                parsed_value_target[i] = 1
            else:
                parsed_value_target[i] += 1

    most_common = {
        'value': 0,
        'name': ''
    }
    
    for i in parsed_value_target:
        if parsed_value_target[i] > most_common['value']:
            most_common['value'] = parsed_value_target[i]
            most_common['name'] = i
    
    return most_common['name']

def most_common_label_target(data,target_attribute):
    most_comm = None
    occ = 0
    for i in data[target_attribute].unique():
        if data[data[target_attribute] == i].shape[0] >  occ:
            most_comm = i
            occ = data[data[target_attribute] == i].shape[0]
    return most_comm

def id3(data, target_attribute, is_IG):
    node = Node()
    if data[target_attribute].nunique()==1:
        node.setLabel(data[target_attribute].unique()[0])
        return node
     
    elif len(play_tennis.columns)==1:
        node.setLabel(get_most_common_label(data, target_attribute))
        return node
    
    else:
        best_attribute_ = best_attribute(data,target_attribute,is_IG)
        node.setAttribute(best_attribute_)
        for i in data[best_attribute_].unique():
            node.addChildren(i,id3(data.loc[data[best_attribute_]==i],target_attribute,is_IG))
            node.set_most_common_label(most_common_label_target(data,target_attribute))        
            
    return node
    
def print_tree(node,depth):
    if node.label is not None: 
        print("    "*(depth+1) +str(node.label))
    else:
        print("    "*depth + "["+ node.attribute +"]")
        for i in node.children:
            print("----"*(depth+1) +str(i))
            print_tree(node.children[i],depth+1)  

def copy_tree(node):
    temp_node = Node()
    if node.label is not None: 
        temp_node.setLabel(node.label)
    else:
        temp_node.setAttribute(node.attribute)
        for i in node.children:
            temp= Node()
            temp= node.children[i]
            temp_node.addChildren(i, temp)
    return temp_node

def check_tree(node,data,index,result, target_attribute):
    if node.label is not None: 
        result.append(node.getLabel())
    else:
        if data.loc[index, node.attribute] is None:
            result.append(node.get_most_common_label())
        for i in node.children:
            if i==data.loc[index,node.attribute]:
                check_tree(node.children[i],data,index,result, target_attribute)

def pred(data,model,target_attribute):
    result = []
    for i in range(len(data)):
        check_tree(model,data[i:i+1],i,result, target_attribute)
    data = {target_attribute:result}
    return pd.DataFrame(data)

# hasn't handle after universal entropy
def information_gain_continous(data, gain_attribute, target_attribute):
    gain_result = 0
    save_boundary = -1
    min_entropy = 999999 
    data = data.sort_values(gain_attribute)
    data=data.reset_index().drop('index',axis=1)
    length_data = len(data)
    for i in range(length_data):
        if (i!=length_data-1):
            if (data.loc[i,target_attribute] != data.loc[i+1,target_attribute]):
                temp_boundary = (data.loc[i,gain_attribute] + data.loc[i+1,gain_attribute])/2
                parsed_data_upper = data.loc[data[gain_attribute]>=temp_boundary]
                parsed_data_lower = data.loc[data[gain_attribute]<temp_boundary]
                len_parsed = len(parsed_data_upper)               
                temp_entropy = float(len_parsed)/length_data * entropy(parsed_data_upper, target_attribute) + float((length_data-len_parsed))/length_data * entropy(parsed_data_lower, target_attribute)     
                if temp_entropy < min_entropy:
                    #print(gain_attribute, temp_boundary, temp_entropy)
                    min_entropy = temp_entropy 
                    save_boundary = temp_boundary

    gain_result += entropy(data,target_attribute) + (-1 * min_entropy)
    return gain_result,save_boundary
    
def best_attribute_c45(data,target_attribute,is_IG):
    gain_attribute = {
        'value': 0,
        'name': '',
        'boundary': -99999999
    }
    
    
    for i in data.columns:
        if (i != target_attribute):
            if is_IG:
                if data[i].dtypes in [pd.np.dtype('float64'), pd.np.dtype('float32')]:
                    ig = information_gain_continous(data, i, target_attribute) 
                    if ig[0] > gain_attribute['value']:
                            gain_attribute['value'] = ig[0]
                            gain_attribute['name'] = i
                            gain_attribute['boundary'] = ig[1]
                else:
                    if information_gain(data, i, target_attribute) > gain_attribute['value']:
                            gain_attribute['value'] = information_gain(data, i, target_attribute)
                            gain_attribute['name'] = i
            else:
                if data[i].dtypes in [pd.np.dtype('float64'), pd.np.dtype('float32')]:
                    ig = gain_ratio_continous(data, i, target_attribute) 
                    if ig[0] > gain_attribute['value']:
                            gain_attribute['value'] = ig[0]
                            gain_attribute['name'] = i
                            gain_attribute['boundary'] = ig[1]
                else:
                    if gain_ratio(data, i, target_attribute) > gain_attribute['value']:
                            gain_attribute['value'] = gain_ratio(data, i, target_attribute)
                            gain_attribute['name'] = i

    return gain_attribute['name'],round(gain_attribute['boundary'],2)

def c45_prun(data, target_attribute,is_IG,vertex_=None):
    node = Node(vertex = vertex_)
    if data[target_attribute].nunique()==1:
        node.setLabel(data[target_attribute].unique()[0])
        return node
     
    elif len(play_tennis.columns)==1:
        node.setLabel(get_most_common_label(data, target_attribute))
        return node
    
    else:
        best_attribute_,bound  = best_attribute_c45(data,target_attribute,is_IG)
        #print(best_attribute_)
        node.setAttribute(best_attribute_)
        node.set_most_common_label(most_common_label_target(data,target_attribute))
        if bound==-99999999:
            for i in data[best_attribute_].unique():
                node.addChildren(i,c45_prun(data.loc[data[best_attribute_]==i],target_attribute,is_IG,i))
        else:
            node.addChildren('>='+str(bound),c45_prun(data.loc[data[best_attribute_]>=bound],target_attribute,is_IG,'>='+str(bound)))
            node.addChildren('<'+str(bound),c45_prun(data.loc[data[best_attribute_]<bound],target_attribute,is_IG,'<'+str(bound)))
    return node

def check_tree_c45(node,data,index,result, target_attribute):
    if node.label is not None: 
        result.append(node.getLabel())
    else:
        if data.loc[index, node.attribute] is None:
            result.append(node.get_most_common_label())
        else:
            for i in node.children:
                #print(i)
                if i[0]=='<':
                    #print(i,2)
                    bound=float(i[1:])
                    if bound>data.loc[index,node.attribute]:
                        #print(data.loc[index,node.attribute])
                        check_tree_c45(node.children[i],data,index,result, target_attribute)
                elif i[0]=='>':
                    bound=float(i[2:])
                    #print(i,1)
                    if bound<=data.loc[index,node.attribute]:
                        #print(data.loc[index,node.attribute])
                        check_tree_c45(node.children[i],data,index,result, target_attribute)
                else:
                    if i==data.loc[index,node.attribute]:
                        check_tree(node.children[i],data,index,result, target_attribute)
                    
def pred_c45(data,model,target_attribute):
    result = []
    for i in range(len(data)):
        check_tree_c45(model,data[i:i+1],i,result, target_attribute)
    data = {target_attribute:result}
    return pd.DataFrame(data)

def accuracy(pred,data):
    cnt = 0
    for i in range(len(pred)):
        if pred.loc[i] == data.loc[i]:
            cnt+=1
    return cnt*100/len(pred)

def get_data_validate(data):
    data_column = data.columns
    data_20 = pd.DataFrame(columns=data_column)
    data_80 = pd.DataFrame(columns=data_column)
    print(data_column)
    count_20 = 0
    count_80 = 0
    
    for i in range(data.shape[0]):
        if(i%5 == 1):
            # Pass the row elements as key value pairs to append() function 
            data_20 = data_20.append(data.loc[[i]] , ignore_index=True)
        else:
            data_80 = data_80.append(data.loc[[i]] , ignore_index=True)
    return data_20, data_80

def prune_tree(node, attribute, vertex):
    if(node.label is not None):
        return node,0
    elif(node.attribute == attribute and node.vertex == vertex):
        node.children = {}
        node.attribute = None
        node.label = node.most_common_label
        
        return node,1
    else:
        for i in node.children:
            a,b = prune_tree(node.children[i], attribute, vertex)
            
        return node,b

def post_pruning(data, node, current_node, children_node, target_attribute):
    if children_node == 0:
        return node
    else:
        for i in current_node.children:
            if current_node.vertex is None: 
                #print(i,1)
                post_pruning(data, node, current_node.children[i], children_node-1, target_attribute)
            else:
                if current_node.attribute is not None:
                    data20, data80 = get_data_validate(data)
                    #print(current_node.attribute,2)
                    save = copy_tree(current_node)
                    temp_node,c = prune_tree(node, current_node.attribute, current_node.vertex)
                    train=data20.drop(target_attribute,axis=1)
                    #print_tree(node,0)
                    temp_node_pred = pred_c45(train, node, target_attribute)
                    #print(temp_node_pred)
                    temp_node_accuracy = accuracy(temp_node_pred[target_attribute], data20[target_attribute])
                    #print(temp_node_accuracy)
                    if temp_node_accuracy == 100:
                        post_pruning(data, temp_node, current_node, children_node, target_attribute)
                    else :
                        
                        node.addChildren(current_node.vertex, save)
                        #print_tree(node,0)


                        post_pruning(data, node, save.children[i], children_node, target_attribute)
        return node
 

def pilihan_model():
    print('Masukkan pilihan dataset : ')
    print('1. Id3Estimator')
    print('2. C45')

def main():
    print('Masukkan pilihan dataset : ')
    print('1. Play_tennis')
    print('2. Iris')
    print('Masukkan pilihan : ')
    pilihan_dataset = input()

    print('Masukkan pilihan measure : ')
    print('1. Information Gain')
    print('2. Gain Ratio')
    print('Masukkan pilihan : ')
    pilihan_measure = input()
    
    measure = True
    if(pilihan_measure=="2"):
        measure = False

    if(pilihan_dataset=="1"):
        pilihan_model()
        print('Masukkan pilihan : ')
        pilihan_model_ = input()
        if(pilihan_model_=="1"):
            model = id3(play_tennis, 'play', measure)
            print_tree(model, 0)
        elif(pilihan_model_=="2"):
            print('Mau di pruning ga? : ')
            print('1. Ya')
            print('2. Tidak')
            pilihan_prun = input()
            if(pilihan_prun=="1"):
                model = c45_prun(play_tennis, "play",measure)
                model_prun = post_pruning(play_tennis, model,model, len(model.children)+1, 'play')
                print_tree(model_prun,0)
            else:
                model = c45_prun(play_tennis, "play",measure)
                print_tree(model,0)

    elif(pilihan_dataset=="2"):
        pilihan_model()
        print('Masukkan pilihan : ')
        pilihan_model_ = input()
        if(pilihan_model_=="1"):
            model = id3(iris, 'target', measure)
            print_tree(model, 0)
        elif(pilihan_model_=="2"):
            print('Mau di pruning ga? : ')
            print('1. Ya')
            print('2. Tidak')
            pilihan_prun = input()
            if(pilihan_prun=="1"):
                model = c45_prun(iris, "target",measure)
                model_prun = post_pruning(iris, model,model, len(model.children)+1, 'target')
                print_tree(model_prun,0)
            else:
                model = c45_prun(iris, "target",measure)
                print_tree(model,0)


if __name__== "__main__":
  main()