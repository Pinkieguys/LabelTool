import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import spam
from labeltool import segmentation
import matplotlib.pyplot as plt
import spam.kalisphera
import numpy


if __name__ == "__main__":
    # First create a seed image, this will be used to induce the segmentation problems
    seeds = spam.kalisphera.kalisphera.makeBlurryNoisySphere(
        (100, 100, 150),
        [[50, 50, 30], [50, 50, 50], [50, 50, 60], [50, 50, 90]],
        [15, 5, 5, 15],
    )
    # Label the seeds
    imSeedLab = spam.label.watershed(seeds > 0.5)
    # Create the binary of the "true" particles
    imGrey = spam.kalisphera.kalisphera.makeBlurryNoisySphere(
        (100, 100, 150),
        [[50, 50, 30], [50, 50, 60], [50, 50, 90], [50, 50, 120]],
        [15, 15, 15, 15],
    )
    # Labels the image and use the seeds, since there is an offset between the seeds and the true image, we will have segmentation problems
    imLabel = spam.label.watershed(imGrey > 0.5, markers=imSeedLab)
    # Let's take a look at it
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title("Label image")
    plt.imshow(imLabel[50, :, :], cmap=spam.label.randomCmap)
    plt.show()

    # From the previous image we saw that we still need to deal with the oversegmentation. First let's compute the oversegmentation coefficient, and colour each label accordingly.


    #
    overSegCoeff, touchingLabels = segmentation.detect_over_segmentation(imLabel)
    imOverSeg = spam.label.convertLabelToFloat(imLabel, overSegCoeff)
    targetOver = numpy.where(overSegCoeff > 0.5)[0]
    label3 = segmentation.fix_over_segmentation(imLabel, targetOver, touchingLabels, imShowProgress=True, verbose=True)

    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title("Oversegmentation coefficient")
    plt.imshow(imOverSeg[50, :, :], cmap=plt.cm.plasma)
    plt.colorbar()
    plt.show()
