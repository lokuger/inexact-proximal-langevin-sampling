import numpy as np

def main():
    A = np.random.normal(size=(3,3,10))
    V = np.var(A,axis=2)
    i = np.unravel_index(np.argsort(V.flatten())[3*3//2],shape=V.shape)
    print(i)
    print(A[i[0],i[1],:])
    print(A[*i,:])
    

if __name__ == '__main__':
    main()