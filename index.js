const tf = require('@tensorflow/tfjs')

const data = [
    'a',
    'b',
    'c',
    'a',
    'b',
    'c',
    'a',
    'b',
    'c'
]

const CHAR_TO_INDEX = {
    'a': 0,
    'b': 1,
    'c': 2
}

const TIME_STEPS = 4
const OUTPUT_VOCABULARY = 3

const model = tf.sequential()
model.add(tf.layers.lstm({
  units: 16,
  inputShape: [TIME_STEPS, OUTPUT_VOCABULARY],
//   recurrentInitializer: 'glorotNormal',
//   returnSequences: true
}))
// model.add(tf.layers.repeatVector({n: TIME_STEPS + 1}));
// model.add(tf.layers.lstm({
//     units: 16,
//     recurrentInitializer: 'glorotNormal',
//     // returnSequences: true
//   }))

model.add(tf.layers.dense({units: OUTPUT_VOCABULARY, activation: 'softmax'}))
// model.add(tf.layers.timeDistributed(
//     {layer: tf.layers.dense({units: OUTPUT_VOCABULARY})}));

const optimizer = tf.train.rmsprop(0.1)
model.compile({optimizer, loss: tf.losses.softmaxCrossEntropy})


function encodeBatch(sequences, numRows) {
    const numExamples = sequences.length;
    const buffer = tf.buffer([numExamples, numRows, OUTPUT_VOCABULARY]);

    for (let n = 0; n < numExamples; ++n) {
        const exampleIndex = n
      const sequence = sequences[n];
      for (let i = 0; i < sequence.length; ++i) {
          const sequenceIndex = i
        const char = sequence[i];
        buffer.set(1, exampleIndex, sequenceIndex, CHAR_TO_INDEX[char]);
      }
    }
    return buffer.toTensor().as3D(numExamples, numRows, OUTPUT_VOCABULARY);
}

function encodeAnswerBatch(nextChars, numRows) {
    const numExamples = nextChars.length;
    const buffer = tf.buffer([numExamples, OUTPUT_VOCABULARY]);

    for (let n = 0; n < numExamples; ++n) {
        const exampleIndex = n
      const char = nextChars[n];
        buffer.set(1, exampleIndex, CHAR_TO_INDEX[char]);
    }
    return buffer.toTensor().as2D(numExamples, OUTPUT_VOCABULARY);
}


const values = []
const inputSequences = []
const nextCharsInSequence = []
for(let i = 0; i < data.length-TIME_STEPS-1; ++i) {
  inputSequences.push(data.slice(i, i+TIME_STEPS))
  nextCharsInSequence.push(data[i+TIME_STEPS])
}
// console.log(inputValues)
// const xs = tf.tensor3d(inputValues, [TIME_STEPS, 1, OUTPUT_VOCABULARY])
const xs = encodeBatch(inputSequences, TIME_STEPS)

// const ys = tf.tensor2d(data.slice(1, 11), [1, OUTPUT_VOCABULARY])
const ys = encodeAnswerBatch(nextCharsInSequence, TIME_STEPS)

console.log(xs.arraySync())
console.log(ys.arraySync())

model.fit(xs, ys, {epochs: 10}).then(h => {
    console.log(h.history.loss)

    console.log(model.predict(encodeBatch([['a', 'b', 'c', 'a']], 4)).arraySync())
})

