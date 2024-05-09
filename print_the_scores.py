#@title print_the_scores(model, processor, dataset)
# ----------------------------Calculate the real Bleu score--------------------------------------
import matplotlib.pyplot as plt
def print_the_scores(model, processor, testing_dataset):
  import matplotlib.pyplot as plt
  image_cidar_scores = []
  image_bleu_scores = []
  image_rouge_scores = []
  image_cnt = 1
  for id in testing_dataset:

    # print ("Image No: ", image_cnt)
    image = id['image']
    text = id['text']
    caption = Caption(model, processor, image)

    references, predictions = [], []
    references.append(text)
    predictions.append(caption)

    # results = bleu.compute(predictions=predictions, references=references, max_order = 2)
    result = get_the_scores(references, predictions)

    image_cidar_scores.append(result['CIDEr']['score'])
    image_bleu_scores.append(result['BLEU']['score'][0])
    image_rouge_scores.append(result['Rouge']['score'])

    image_cnt = image_cnt + 1

  final_bleu_score = sum(image_bleu_scores) / len(image_bleu_scores)
  final_cidar_score = sum(image_cidar_scores) / len(image_bleu_scores)
  final_rouge_score = sum(image_rouge_scores) / len(image_bleu_scores)

  plt.subplot(3, 1, 1)
  bleu_scores = list(image_bleu_scores)  # Extract all BLEU scores as a list
  plt.scatter(range(len(bleu_scores)), bleu_scores)  # Use range(len()) for x-axis

  # Customization (optional)
  plt.xlabel("Image Index")
  plt.ylabel("BLEU Score")
  # plt.title("BLEU Scores vs. Image Index")
  plt.title("Overall BLEU Score: %.3f" % final_bleu_score)
  plt.grid(True)

  plt.subplot(3, 1, 2)
  cidar_scores = list(image_cidar_scores)  # Extract all BLEU scores as a list
  plt.scatter(range(len(cidar_scores)), cidar_scores)  # Use range(len()) for x-axis

  # Customization (optional)
  plt.xlabel("Image Index")
  plt.ylabel("CIDEr Score")
  plt.title("Overall CIDEr Score: %.3f" % final_cidar_score)
  plt.grid(True)

  # ---------------Code for Rouge plot------------------ #
  plt.subplot(3, 1, 3)
  rouge_scores = list(image_rouge_scores)  # Extract all BLEU scores as a list
  plt.scatter(range(len(rouge_scores)), rouge_scores)  # Use range(len()) for x-axis

  # Customization (optional)
  plt.xlabel("Image Index")
  plt.ylabel("Rouge Score")
  plt.title("Overall Rouge Score: %.3f" % final_rouge_score)
  plt.grid(True)

  # Adjust layout to prevent overlap
  plt.tight_layout()

  # Display the plot
  plt.show()