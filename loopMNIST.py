from MNISTFGSM import executeLoop

dirs = ['MNIST01', 'MNIST05', 'MNIST10', 'MNIST15', 'MNIST20', 'MNIST25', 'MNIST30', 'MNIST35', 'MNIST40', 'MNIST45', 'MNIST50', 'MNIST55', 'MNIST60', 'MNIST65', 'MNIST70', 'MNIST75', 'MNIST80', 'MNIST85', 'MNIST90', 'MNIST95']

attacks = ['fgml', 'mifgml']

for dir in dirs:
    for att in attacks:
        executeLoop(dir, att)
