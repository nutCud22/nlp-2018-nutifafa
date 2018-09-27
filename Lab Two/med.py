
# coding: utf-8

# In[80]:


import sys

def min_edit_distance(source, target):
    n = len(source)
    m = len(target)
    
    distance = [[None for _ in range(m+1)] for _ in range(n+1)]
    distance[0][0] = 0
    
    for i in range(1, n+1):
        distance[i][0] = distance[i-1][0] + 1
    for j in range(1, m+1):
        distance[0][j] = distance[0][j-1] + 1
        
    for i in range(1, n+1):
        substitution_value = 2
        for j in range(1, m+1):
            if source[i-1] == target[j-1]:
                substitution_value = 0
            distance[i][j] = min(distance[i-1][j] + 1, distance[i-1][j-1] + substitution_value, distance[i][j-1] + 1)
            
    print("Minimum edit distance between " + source + " and " + target + " is " + str(distance[n][m]))
    
if __name__ == '__main__':
    source = sys.argv[1]
    target = sys.argv[2]
    min_edit_distance(source, target)

