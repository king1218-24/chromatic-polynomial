import random
import numpy as np
def adjacency_matrix_to_edge_list(matrix):
    n = len(matrix)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 1:
                edges.append((i, j))
    return edges

def find_connected_components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            component = {i}
            while stack:
                node = stack.pop()
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
    return components
#删除某条边，合并两个点
def contract_edge(edges, u, v):
    new_edges = []
    for a, b in edges:
        a_new = a if a != v else u
        b_new = b if b != v else u
        x, y = sorted((a_new, b_new))
        if x>v:
            x=x-1
        if y>v:
            y=y-1
        if x != y:
            new_edges.append((x, y))
    unique_edges = list(dict.fromkeys(new_edges))
    return unique_edges

def multiply_polynomials(p1, p2):
    result = {}
    for d1, c1 in p1.items():
        for d2, c2 in p2.items():
            d = d1 + d2
            c = c1 * c2
            result[d] = result.get(d, 0) + c
    return {d: c for d, c in result.items() if c != 0}

def subtract_polynomials(p1, p2):
    result = p1.copy()
    for d, c in p2.items():
        result[d] = result.get(d, 0) - c
    return {d: c for d, c in result.items() if c != 0}

def chromatic_polynomial(n, edges):
    components = find_connected_components(n, edges)
    if len(components) > 1:
        poly = {0: 1}
        for comp in components:
            comp_n = len(comp)
            licomp=list(comp)
            sorted(licomp)
            dic={}
            for i in range(comp_n):
                dic[licomp[i]]=i
            comp_edges = [(dic[u],dic[v]) for u, v in edges if u in comp and v in comp]

            comp_poly = chromatic_polynomial(comp_n, comp_edges)
            poly = multiply_polynomials(poly, comp_poly)
        return poly
    if not edges:
        return {n: 1}
    else:
        e = edges[0]
        u, v = e
        edges_without_e = [edge for edge in edges if edge != e]
        p1 = chromatic_polynomial(n, edges_without_e)
        contracted_edges = contract_edge(edges, u, v)
        p2 = chromatic_polynomial(n - 1, contracted_edges)
        return subtract_polynomials(p1, p2)

def polynomial_to_string(poly):
    if not poly:
        return "0"
    terms = []
    degrees = sorted(poly.keys(), reverse=True)
    for degree in degrees:
        coeff = poly[degree]
        if coeff == 0:
            continue
        if degree == 0:
            term = f"{coeff}"
        elif degree == 1:
            if coeff==1:
                term="k"
            elif coeff==-1:
                term="-k"
            else:
                term = f"{coeff}k"
        else:
            if coeff==1:
                term=f"k^{degree}"
            elif coeff==-1:
                term=f"-k^{degree}"
            else:
                term = f"{coeff}k^{degree}"
        if coeff > 0:
            term = "+" + term if terms else term
        else:
            term = term if terms else term
        terms.append(term)
    expr = " ".join(terms).replace("+ -", "- ")
    return expr

n=input("the size of the vertices set: ")
n=int(n)
matrix=[[] for _ in range(n)]
print("please input the adjacency matrix")
for i in range(n):
    s=input().split(" ")
    matrix[i]=[int(x) for x in s]
#用于生成指定点数和指定边数的随机图
'''
n=int(input("顶点数:"))
e=int(input("边数:"))
matrix=np.zeros([n,n])
for i in range(e):
    v1=random.randint(0,n-1)
    v2=random.randint(0,n-1)
    while v1==v2 or matrix[v1][v2]==1:
        v1=random.randint(0,n-1)
        v2=random.randint(0,n-1)
    matrix[v1][v2]=1
    matrix[v2][v1]=1
'''

edges = adjacency_matrix_to_edge_list(matrix)
poly = chromatic_polynomial(len(matrix), edges)
print(polynomial_to_string(poly)) 
