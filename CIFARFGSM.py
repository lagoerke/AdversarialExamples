def executeLoop(path, attack):
    # coding: utf-8
    # Many parts of the code are taken or inspired from https://github.com/gongzhitaao/adversarial-classifier and https://github.com/anishathalye/obfuscated-gradients

    # --------------------------
    # ---       Imports      ---
    # --------------------------

    import os
    # supress tensorflow logging other than errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import numpy as np
    np.random.seed(42)
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    from keras import backend as K
    from keras.datasets import cifar10

    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from keras import optimizers
    from keras.utils.generic_utils import get_custom_objects

    from keras.callbacks import LambdaCallback, TerminateOnNaN, EarlyStopping


    # -------------------------
    # ---     Functions     ---
    # -------------------------

    def predict_with_logits(model, input):
        for i, layer in enumerate(model.layers):
            output = input
            input = layer (input)
        return input, output

    # ---------------------------------
    # ---    Adversarial Attack     ---
    # ---------------------------------

    # Different adversarial attacks we can use for creating adversarial images.

    def fgml(model, xadv, eps=0.01, epochs=1, clip_min=0., clip_max=1.):
        """
        :param model: A wrapper that returns the output as well as logits.
        :param x: The input placeholder.
        :param eps: The scale factor for noise.
        :param epochs: The maximum epoch to run.
        :param clip_min: The minimum value in output.
        :param clip_max: The maximum value in output.
        :return: A tensor, contains adversarial samples for each input.
        """
        x = tf.convert_to_tensor(xadv)
        ybar , _ = predict_with_logits(model, x)
        indices = tf.argmax(ybar, axis=1)
        ydim = ybar.get_shape().as_list()[1]
        indices = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)

        for i in range(epochs):
            # determine loss with logits and target
            ybar , logits = predict_with_logits(model, x)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=indices, logits=logits)
            # gradient of the loss when changing xadv
            dy_dx, = tf.gradients(loss, x)
            x = tf.stop_gradient(x + eps*tf.sign(dy_dx))
            # make sure values are between 0 and 1
            x = tf.clip_by_value(x, clip_min, clip_max)
        return x

    def mifgml(model, xadv, vel, eps=16/255, epochs=10, decay = 1, clip_min=0., clip_max=1.):
        """
        :param model: A wrapper that returns the output as well as logits.
        :param x: The input placeholder.
        :param eps: The scale factor for noise.
        :param epochs: The maximum epoch to run.
        :param clip_min: The minimum value in output.
        :param clip_max: The maximum value in output.
        :return: A tensor, contains adversarial samples for each input.
        """
        x = tf.convert_to_tensor(xadv)
        ybar , _ = predict_with_logits(model, x)
        indices = tf.argmax(ybar, axis=1)
        ydim = ybar.get_shape().as_list()[1]
        indices = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)

        for i in range(epochs):
            # determine loss with logits and target
            ybar , logits = predict_with_logits(model, x)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=indices, logits=logits)
            # gradient of the loss when changing xadv
            dy_dx, = tf.gradients(loss, x)
            # velocity vector for added momentum
            vel = decay * vel + dy_dx/tf.norm(dy_dx, ord=1)
            x = tf.stop_gradient(x + eps*tf.sign(vel))
            # make sure values are between 0 and 1
            x = tf.clip_by_value(x, clip_min, clip_max)
        return x


    # ---------------------------
    # ---     Prepare Data    ---
    # ---------------------------

    # Shape of Dataset Cifar10
    img_rows = 32
    img_cols = 32
    img_chan = 3
    input_shape=(img_rows, img_cols, img_chan)
    nb_classes = 10

    # Loading Data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = np.reshape(X_train, (-1, img_rows, img_cols, img_chan))
    X_test = np.reshape(X_test, (-1, img_rows, img_cols, img_chan))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    print('\nX_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)


    # -------------------------------
    # ---     Starting Session    ---
    # -------------------------------

    sess = tf.InteractiveSession()
    K.set_session(sess)


    # ---------------------
    # ---     Model     ---
    # ---------------------

    print('\nLoading model0')
    pathmod0 = path + '/model_test.h5'
    model0 = load_model(pathmod0)


    # ----------------------
    # ---     Attack     ---
    # ----------------------

    x = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chan))
    eps = tf.placeholder(tf.float32, ())
    vel = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chan))

    EPS = 0.005

    if (attack == 'fgml'):
        x_adv = fgml(model0, x, epochs=9, eps=eps)
    elif (attack =='mifgml'):
        x_adv = mifgml(model0, x, vel, epochs=9, eps=eps)
    else:
        raise Exception("Attack not supported")


    nb_sample = X_train.shape[0]
    batch_size = 128
    nb_batch = int(np.ceil(nb_sample/batch_size))

    print('\nBuilding X_test_adv')
    nb_sample = X_test.shape[0]
    nb_batch = int(np.ceil(nb_sample/batch_size))
    X_test_adv = np.empty(X_test.shape)
    for batch in range(nb_batch):
        print(' batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
        start = batch * batch_size
        end = min(nb_sample, start+batch_size)
        tmp = sess.run(x_adv, feed_dict={x: X_test[start:end],
                                         eps: EPS,
                                         vel: np.zeros(X_test[start:end].shape),
                                         K.learning_phase(): 0})
        X_test_adv[start:end] = tmp
    print('\nSaving adversarial images')
    pathadv0 = path + '/adv01cifar' + attack + '_{0:.4f}.npz'.format(EPS)
    np.savez(pathadv0,
             X_test_adv=X_test_adv)


    print('\nTesting against adversarial test data')
    score = model0.evaluate(X_test_adv, y_test)
    print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


    # -------------------------------
    # ---    Distilled  Model     ---
    # -------------------------------


    print('\nLoading model1')
    pathmod1 = path + '/model_test2.h5'
    model1 = load_model(pathmod1)


    # ----------------------
    # ---     Attack     ---
    # ----------------------

    x = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chan))
    eps = tf.placeholder(tf.float32, ())
    vel = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chan))

    EPS = 0.005

    if (attack == 'fgml'):
        x_adv2 = fgml(model1, x, epochs=9, eps=eps)
    elif (attack =='mifgml'):
        x_adv2 = mifgml(model1, x, vel, epochs=9, eps=eps)
    else:
        raise Exception("Attack not supported")

    nb_sample = X_train.shape[0]
    batch_size = 128
    nb_batch = int(np.ceil(nb_sample/batch_size))

    print('\nBuilding X_test_adv')
    nb_sample = X_test.shape[0]
    nb_batch = int(np.ceil(nb_sample/batch_size))
    X_test_adv2 = np.empty(X_test.shape)
    for batch in range(nb_batch):
        print(' batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
        start = batch * batch_size
        end = min(nb_sample, start+batch_size)
        tmp = sess.run(x_adv2, feed_dict={x: X_test[start:end],
                                         eps: EPS,
                                         vel: np.zeros(X_test[start:end].shape),
                                         K.learning_phase(): 0})
        X_test_adv2[start:end] = tmp

    print('\nSaving adversarial images')
    pathadv1 = path + '/adv02cifar02' + attack + '_{0:.4f}.npz'.format(EPS)
    np.savez(pathadv1,
             X_test_adv=X_test_adv2)


    print('\nTesting against adversarial test data')
    score = model1.evaluate(X_test_adv, y_test)
    print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


    # -----------------------------
    # ---     Visualization     ---
    # -----------------------------

    print('\nPlotting random adversarial data')


    print('Accuracy Model1')
    scored = model0.evaluate(X_test, y_test, verbose=1)
    print('Test 1 loss:', scored[0])
    print('Test 1 accuracy:', scored[1])
    print('Accuracy Model1 Adv1')
    scoree = model0.evaluate(X_test_adv, y_test, verbose=1)
    print('Test 1 loss:', scoree[0])
    print('Test 1 accuracy:', scoree[1])
    print('Accuracy Model1 Adv2')
    scoref = model0.evaluate(X_test_adv2, y_test, verbose=1)
    print('Test 1 loss:', scoref[0])
    print('Test 1 accuracy:', scoref[1])


    print('Accuracy Model2')
    scorea = model1.evaluate(X_test, y_test, verbose=1)
    print('Test 1 loss:', scorea[0])
    print('Test 1 accuracy:', scorea[1])
    print('Accuracy Model2 Adv1')
    scoreb = model1.evaluate(X_test_adv, y_test, verbose=1)
    print('Test 1 loss:', scoreb[0])
    print('Test 1 accuracy:', scoreb[1])
    print('Accuracy Model2 Adv2')
    scorec = model1.evaluate(X_test_adv2, y_test, verbose=1)
    print('Test 1 loss:', scorec[0])
    print('Test 1 accuracy:', scorec[1])

    pathtxt = path + '/accuracy_{0:.4f}.txt'.format(EPS)
    with open(pathtxt, 'w') as text_file:
        print('Accuracy Model1', file=text_file)
        print('Test 1 loss:', scored[0], file=text_file)
        print('Test 1 accuracy:', scored[1], file=text_file)
        print('Accuracy Model1 Adv1', file=text_file)
        print('Test 2 loss:', scoree[0], file=text_file)
        print('Test 2 accuracy:', scoree[1], file=text_file)
        print('Accuracy Model1 Adv2', file=text_file)
        print('Test 3 loss:', scoref[0], file=text_file)
        print('Test 3 accuracy:', scoref[1], file=text_file)


        print('Accuracy Model2', file=text_file)
        print('Test 1 loss:', scorea[0], file=text_file)
        print('Test 1 accuracy:', scorea[1], file=text_file)
        print('Accuracy Model2 Adv1', file=text_file)
        print('Test 2 loss:', scoreb[0], file=text_file)
        print('Test 2 accuracy:', scoreb[1], file=text_file)
        print('Accuracy Model2 Adv2', file=text_file)
        print('Test 3 loss:', scorec[0], file=text_file)
        print('Test 3 accuracy:', scorec[1], file=text_file)



    print('\nMaking predictions')
    print('Labels')
    z0 = np.argmax(y_test, axis=1)
    print("z0: ",z0)

    print('Erstes Model Original Bilder')
    y6 = model0.predict(X_test)
    z6 = np.argmax(y6, axis=1)
    print("z0: ",z6)
    print('Erstes Model erste adversarial Bilder')
    y4 = model0.predict(X_test_adv)
    z4 = np.argmax(y4, axis=1)
    print("z4: ",z4)
    print('Erste Model zweite adversarial Bilder')
    y5 = model0.predict(X_test_adv2)
    z5 = np.argmax(y5, axis=1)
    print("z5: ",z5)

    print('Zweites Model Original Bilder')
    y1 = model1.predict(X_test)
    z1 = np.argmax(y1, axis=1)
    print("z1: ",z1)
    print('Zweites Model erste adversarial Bilder')
    y2 = model1.predict(X_test_adv)
    z2 = np.argmax(y2, axis=1)
    print("z2: ",z2)
    print('Zweites Model zweite adversarial Bilder')
    y3 = model1.predict(X_test_adv2)
    z3 = np.argmax(y3, axis=1)
    print("z3: ",z3)


    print("falsch 1:")
    ind3 = np.where(z0!=z6)
    print(ind3[0])
    print("Länge")
    print(len(ind3[0]))

    print("falsch adv1 1:")
    ind3a = np.where(z0!=z4)
    print(ind3a[0])
    print("Länge")
    print(len(ind3a[0]))

    print("falsch adv2 1:")
    ind3b = np.where(z0!=z5)
    print(ind3b[0])
    print("Länge")
    print(len(ind3b[0]))

    print("falsch 2:")
    ind2 = np.where(z1!=z0)
    print(ind2[0])
    print("Länge")
    print(len(ind2[0]))

    print("falsch adv1 2:")
    ind2a = np.where(z0!=z2)
    print(ind2a[0])
    print("Länge")
    print(len(ind2a[0]))

    print("falsch adv2 2:")
    ind2b = np.where(z3!=z0)
    print(ind2[0])
    print("Länge")
    print(len(ind2b[0]))

    print("ungleich:")
    ind4 = np.where(z1!=z6)
    print(ind4[0])
    print("Länge")
    print(len(ind4[0]))

    skip = True

    print('\nSelecting figures')
    X_tmp = np.empty((6, nb_classes, img_rows, img_cols, img_chan))
    y_proba = np.empty((6, nb_classes, nb_classes))
    for i in range(10):
        print('Target {0}'.format(i))
        ind, = np.where(np.all([z0==i, z1==i, z2!=i, z3!=i, z6==i], axis=0))
        print("ind ", i, " is ", ind)
        if not len(ind):
            skip = False
            break
        cur = np.random.choice(ind)
        X_tmp[0][i] = np.squeeze(X_test[cur])
        X_tmp[1][i] = np.squeeze(X_test_adv[cur])
        X_tmp[2][i] = np.squeeze(X_test_adv2[cur])
        X_tmp[3][i] = np.squeeze(X_test[cur])
        X_tmp[4][i] = np.squeeze(X_test_adv[cur])
        X_tmp[5][i] = np.squeeze(X_test_adv2[cur])
        y_proba[0][i] = y1[cur]
        y_proba[1][i] = y2[cur]
        y_proba[2][i] = y3[cur]
        y_proba[3][i] = y6[cur]
        y_proba[4][i] = y4[cur]
        y_proba[5][i] = y5[cur]


    # If no examples exist for some class, skip the plotting so that the overall loop will not crash
    if skip:
        print('\nPlotting results')
        fig = plt.figure(figsize=(2, 10))
        gs = gridspec.GridSpec(2, 10, wspace=0.1, hspace=0.1)

        label = np.argmax(y_proba, axis=2)
        proba = np.max(y_proba, axis=2)
        for i in range(10):
            for j in range(2):
                ax = fig.add_subplot(gs[j, i])
                ax.imshow(X_tmp[j][i], interpolation='none')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('{0} ({1:.2f})'.format(label[j][i],
                                             proba[j][i]),
                                             fontsize=12)

        print('\nSaving figure')
        gs.tight_layout(fig)
        pathfig = path + '/cifar_pics_{0:.4f}.pdf'.format(EPS)
        plt.savefig(pathfig)
