from CIFARFGSM import executeLoop

dirs = ['CIFAR01', 'CIFAR05', 'CIFAR10', 'CIFAR15', 'CIFAR20','CIFAR25', 'CIFAR30', 'CIFAR35', 'CIFAR45', 'CIFAR50','CIFAR55', 'CIFAR60', CIFAR65', 'CIFAR70', 'CIFAR75', 'CIFAR80', 'CIFAR85', 'CIFAR90',  'CIFAR95']

attacks = ['fgml', 'mifgml']

for dir in dirs:
    for att in attacks:
        executeLoop(dir, att)
