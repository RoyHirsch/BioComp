model = Sequential()
model.add(Conv1D(32, kernel_size=7,activation='relu',input_shape=[36,4]))
model.add(Conv1D(64, 7, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit_generator(generator=dataPipe.train_generator,
                    validation_data=None,
                    steps_per_epoch=6000,
                    use_multiprocessing=True,
                    workers=6,
                    verbose=1)

model.predict_generator(generator=dataPipe.test_generator,
                    use_multiprocessing=True,
                    workers=6,
                    verbose=1)
