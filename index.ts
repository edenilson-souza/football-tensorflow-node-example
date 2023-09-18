import * as tf from "@tensorflow/tfjs-node";
import { tidy, scalar, tensor2d, Tensor2D } from "@tensorflow/tfjs-node";

// Fazer previsões
// async function predictOutcome(newMatch: { homeScore: number; awayScore: number }) {
//     const xNew = tensor2d([[newMatch.homeScore, newMatch.awayScore]]);
//     const prediction = await tidy(() => model.predict(xNew) as Tensor2D);

//     const predictedResult = prediction.dataSync()[0] > 0.5 ? "Win" : "Loss";
//     console.log(`Resultado previsto para a nova partida: ${predictedResult}`);
// }
// // Treinar o modelo e fazer previsões
// trainModel().then(() => {
//     const newMatch = { homeScore: 25, awayScore: 20 };
//     predictOutcome(newMatch);
// });

// Carregar modelo e fazer previsões
tf.loadLayersModel("file://./model/model.json").then(loadedModel => {
    const newMatch: any[] = [
        { homeScore: 15, awayScore: 21 }, //Loss
        { homeScore: 21, awayScore: 15 }, //Win
        { homeScore: 15, awayScore: 15 }, //Loss
        { homeScore: 15, awayScore: 25 }, //Loss
        { homeScore: 25, awayScore: 15 }, //Win
        { homeScore: 15, awayScore: 16 }, //Loss
        { homeScore: 16, awayScore: 15 }, //Win
        { homeScore: 1, awayScore: 2 }, //Loss
        { homeScore: 1, awayScore: 0 }, //Win
        { homeScore: 1, awayScore: 1 } //Loss
    ];

    newMatch.forEach(match => {
        const xNew = tensor2d([[match.homeScore, match.awayScore]]);
        const prediction = tidy(() => loadedModel.predict(xNew) as Tensor2D);

        const predictionTeam1 = prediction.dataSync()[0] * 100;

        const predictedResult = prediction.dataSync()[0] > 0.5 ? "Win" : "Loss";
        const odd = {
            home: predictionTeam1,
            away: 100 - predictionTeam1
        };

        console.log(`${predictedResult}: ${odd.home.toFixed(2)}% - ${odd.away.toFixed(2)}%`);
    });
});
