import * as tf from "@tensorflow/tfjs-node";
import { tensor2d } from "@tensorflow/tfjs-node";

// Dados históricos de partidas de futebol
const matches = [
    { homeScore: 32, awayScore: 30, result: "Win" }, //Adicione mais estatísticas aqui...
    { homeScore: 30, awayScore: 32, result: "Loss" },
    { homeScore: 30, awayScore: 30, result: "Loss" },
    { homeScore: 30, awayScore: 40, result: "Loss" },
    { homeScore: 20, awayScore: 15, result: "Win" },
    { homeScore: 15, awayScore: 20, result: "Loss" },
    { homeScore: 20, awayScore: 20, result: "Loss" },
    { homeScore: 26, awayScore: 25, result: "Win" },
    { homeScore: 25, awayScore: 26, result: "Loss" },
    { homeScore: 25, awayScore: 25, result: "Loss" },
    { homeScore: 1, awayScore: 2, result: "Loss" },
    { homeScore: 1, awayScore: 0, result: "Win" },
    { homeScore: 1, awayScore: 1, result: "Loss" },
    { homeScore: 2, awayScore: 1, result: "Win" },
    { homeScore: 0, awayScore: 1, result: "Loss" },
    { homeScore: 1, awayScore: 1, result: "Loss" },
    { homeScore: 1, awayScore: 2, result: "Loss" },
    { homeScore: 3, awayScore: 0, result: "Win" },
    { homeScore: 1, awayScore: 0, result: "Win" },
    { homeScore: 2, awayScore: 0, result: "Win" },
    { homeScore: 0, awayScore: 1, result: "Loss" },
    { homeScore: 0, awayScore: 0, result: "Loss" },
    { homeScore: 0, awayScore: 2, result: "Loss" }
    // Adicione mais partidas aqui...
];

// Preparando os dados
const inputData: number[][] = [];
const outputData: number[] = [];

for (const match of matches) {
    const homeScore = match.homeScore;
    const awayScore = match.awayScore;
    //Preparando os dados de entrada e saída
    //Se adicionar mais estatísticas, adicione mais valores aqui
    const result = match.result === "Win" ? 1 : 0; // Codificar 'Win' como 1 e 'Loss' como 0

    inputData.push([homeScore, awayScore]); // Adicionando os dados de entrada
    outputData.push(result); // Adicionando os dados de saída
}

const xData = tensor2d(inputData);
const yData = tensor2d(outputData, [outputData.length, 1]);

// Criando o modelo de Regressão Logística
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2], activation: "sigmoid" }));

// Compilando o modelo
model.compile({ optimizer: tf.train.adam(), loss: "binaryCrossentropy", metrics: ["accuracy"] });

// Treinando o modelo
async function trainModel() {
    const epochsTotal = 10000;
    // const epochsPerBatch = 10;
    // const batchSize = 4;
    const verbose = 1;
    await model
        .fit(xData, yData, {
            epochs: epochsTotal,
            verbose: verbose,
            // batchSize: batchSize,
            // stepsPerEpoch: Math.ceil(xData.shape[0] / batchSize) * epochsPerBatch,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch: any, logs: any) => {
                    console.log(`Epoch ${epoch + 1}/${epochsTotal} - loss: ${logs?.loss.toFixed(4)} - acc: ${logs?.acc.toFixed(4)}`);
                }
            }
        })
        .then(() => {
            console.log("Treinamento concluído.");
            model.summary();
            //SAVE
            const trainDate = new Date();
            model.save(`file://./${trainDate.getTime()}`).then(() => {
                console.log("Modelo salvo.");
            });
        });
}

trainModel();
