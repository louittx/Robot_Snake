import torch

x = torch.tensor(5.0,requires_grad=True) # mettre les .0 pour deffiner le float sino cela ne focntionne plus
Y = (x**3)-((4*x)**2)+(2*x)
Y.backward()
print(x.grad)

x2 = torch.tensor(5.0,requires_grad=True)
y2 = torch.tensor(5.0,requires_grad=True)

F = ((x2**2)*y2)+(y2**3)
F.backward()
print(x2.grad)
print(y2.grad)

x3 = torch.tensor(2.0, dtype=torch.float32,requires_grad=True)
a = 0.01
for i in range(1000):
    Y2 = (2*x3)+1
    Y2.backward()
    with torch.no_grad():
        x3 -= a*(x3.grad) # oblicatoir sino il va plus en torch.tensor
    x3.grad.zero_()
print(x3)

#Sur torch il fonction comme Numpy avec les méme fonction 
A = torch.arange(1,10).reshape(3, 3) # crées un vecteur lineaire de 1 à 9 puis le réajuste en matrcie 3 3
B = torch.full((3,3),2)
print('A = ',A,'\n')
B = A @ B
print('B = ',B,'\n')

x = ((torch.rand(4, 4) * 180).float())-90
print(x)