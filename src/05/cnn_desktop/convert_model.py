import tensorflow as tf

def loadModel():
    return tf.keras.models.load_model('mnist_model.h5')

if __name__ == '__main__':
    model = loadModel()
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.save('mnist_model.keras')
