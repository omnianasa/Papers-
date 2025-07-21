## ANeural Algorithm of Artistic Style

This paper introduces a new way to mix the content of one image with the style of another by using a deep neural network trained for object recognition (like VGG-19). This approach manipulates features extracted from different layers of the network unlike dealing with pixels like old methods.


### Important Parameters

The neural network learns two different things:
- **Content Representation**: What’s in the image (objects, structure, layout) and it is captured by the activation values in deeper layers
- **Style Representation**: How the image looks (colors, textures, brushstrokes) and it is captured by correlations between feature maps

| Parameter         | Description                                                      | Typical Values          |
|------------------|------------------------------------------------------------------|--------------------------|
| `content_weight` | How much we care about content vs style.                         | 1e4 (or 1e5 for sharper content) |
| `style_weight`   | How much we care about matching the style features.              | 1e-2 to 1e2              |
| `num_steps`      | Number of iterations for optimization.                           | 300 - 1000               |
| `learning_rate`  | Step size for updating the image.                                | 0.003 - 0.1              |
| `style_layers`   | Layers used to compute style features (usually from shallow layers). | ['conv1_1', 'conv2_1', ...] |
| `content_layers` | Layers used to compute content features (usually deep layers).   | ['conv4_2']              |


### How to Benefit from It

- It’s a great way to understand how deep networks see images.
- Can be used in art, filmmaking
- Help artists to create new art forms.
- Researchers to explore how the human visual system might separate content and appearance.
- Developers to build creative tools and filters.


