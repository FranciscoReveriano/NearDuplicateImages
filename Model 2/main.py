# Predict
# example_filename = test_filenames[0[
example_filename = train_filenames[3]
print(example_filename)
#Function_Library.view_image(example_filename)
example_img = image.load_img(example_filename, target_size=(IMG_WIDTH, IMG_HEIGHT))
example_img = image.img_to_array(example_img)
example_img = np.expand_dims(example_img, axis=0)
example_img = preprocess_input(example_img)
prediction = joined_model.predict([example_img]).reshape(-1)

img = ''
for index in prediction.argsort()[-2:][::-1]:
    img = train_filenames[index]
    if img == example_filename:
        continue
    #Function_Library.view_image(img)
    print(train_filenames[index])

# Display Both Images
Function_Library.view_both_images(example_filename, img)