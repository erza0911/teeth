import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_oct_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: normal or abnormal
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_data(train_dir, test_dir, input_shape=(64, 64, 1)):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale'
    )

    return train_generator, test_generator

def train_oct_model(model, train_generator, validation_generator, epochs=10):
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

def evaluate_oct_model(model, test_generator):
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')

def predict_and_visualize(model, image_path, input_shape=(64, 64, 1), threshold=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_for_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    if prediction > threshold:
        print("该图像为异常（蛀牙或裂纹）")

        # 添加标记
        contours, _ = cv2.findContours((img[0] > threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_for_display, contours, -1, (0, 0, 255), 2)  # 在图像上绘制红色轮廓
    else:
        print("该图像为正常")

    # 显示结果
    cv2.imshow("Original Image", img_for_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_data_dir = 'path/to/train_data'
    test_data_dir = 'path/to/test_data'

    input_shape = (64, 64, 1)
    model = create_oct_model(input_shape)

    train_generator, test_generator = prepare_data(train_data_dir, test_data_dir, input_shape)
    train_oct_model(model, train_generator, test_generator, epochs=10)
    evaluate_oct_model(model, test_generator)

    # 用测试图片进行预测并可视化
    test_image_path = 'path/to/test_image.jpg'
    predict_and_visualize(model, test_image_path, input_shape)
