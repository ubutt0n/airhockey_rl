#include <iostream>
#include <fstream>
#include <SDL.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <ranges>
#include <iterator>

using namespace std;

const int WIDTH = 600, HEIGHT = 800, PADDLE_WIDTH = 80, PADDLE_HEIGHT = 20, PADDLE_SPEED_1 = 6, PADDLE_SPEED_2 = 6, PUCK_RADIUS = 13, PUCK_SPEED_X = 4, PUCK_SPEED_Y = 4;
const int DataSetMaxSize = 100000;

struct game_step
{
    Eigen::MatrixXd state;
    Eigen::MatrixXd next_state;
    int reward;
    int action;
    bool done;
};

class DataSet
{
protected:
    vector<game_step> data;
    int count;
public:
    void store_data(const game_step& iteration)
    {
        if (count == DataSetMaxSize)
        {
            data.erase(data.begin());
            count--;
        }
        data.push_back(iteration);
        count++;
    }

    vector<game_step> get_batch(int batch_size)
    const {
        vector<game_step> res;
        auto gen = mt19937{ random_device{}() };
        std::ranges::sample(data, std::back_inserter(res), batch_size, gen);

        return res;
    }

    int get_count()
    const {
        return count;
    }
};

class Layer
{
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& dprev, double lr = 0.01, bool train = true) = 0;
};

class Linear : public Layer
{
protected:
    Eigen::MatrixXd W;
    Eigen::MatrixXd X;
    Eigen::MatrixXd b;
public:
    Linear(int in_features, int out_features)
    {
        W = Eigen::MatrixXd::Random(in_features, out_features);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override
    {
        X = x;

        return (X * W);
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& dprev, double lr = 0.001, bool train = true) override
    {
        Eigen::MatrixXd res = dprev * W.transpose();
        W = train ? W - (lr * (X.transpose() * dprev)) : W;

        return res;
    }
};

class ReLU : public Layer
{
protected:
    Eigen::MatrixXd resX;
public:
    ReLU() {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override
    {
        Eigen::MatrixXd res(x.rows(), x.cols());
        for (int r = 0; r < x.rows(); r++)
        {
            for (int c = 0; c < x.cols(); c++)
            {
                res(r, c) = x(r, c) > 0 ? x(r, c) : 0;
            }
        }
        resX = res;

        return res;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& dprev, double lr = 0.01, bool train = true) override
    {
        Eigen::MatrixXd res(resX.rows(), resX.cols());
        for (int r = 0; r < resX.rows(); r++)
        {
            for (int c = 0; c < resX.cols(); c++)
            {
                res(r, c) = resX(r, c) > 0 ? dprev(r, c) : 0;
            }
        }
        return res;
    }
};

class MSE
{
protected:
    Eigen::MatrixXd y;
    Eigen::MatrixXd y_p;
public:
    MSE() {}

    double forward(const Eigen::MatrixXd& y1, const Eigen::MatrixXd& y_p1)
    {
        y = y1;
        y_p = y_p1.transpose();

        return ((y - y_p).array() * (y - y_p).array()).sum() / (y.cols() * y.rows());
    }

    Eigen::MatrixXd backward(double dprev)
    {
        return (-2 * (y - y_p) / (y.cols() * y.rows())).transpose();
    }
};

class Sequence
{
protected:
    std::vector<Layer*> sq;
public:
    Sequence() {}
    Sequence(vector<Layer*> arr) : sq(arr) {}

    void append(Layer* layer)
    {
        sq.push_back(layer);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x)
    {
        Eigen::MatrixXd res;
        Eigen::MatrixXd x1 = x;

        for (std::vector<Layer*>::iterator i = sq.begin(); i != sq.end(); i++)
        {
            res = (*i)->forward(x1);
            x1 = res;
        }

        return res;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& dprev)
    {
        Eigen::MatrixXd dprev1 = dprev;

        std::vector<Layer*>::iterator i = sq.end();
        i--;

        for (i; i != sq.begin(); i--)
        {
            dprev1 = (*i)->backward(dprev1);
        }

        return dprev1;
    }
};

void DrawCircle(SDL_Renderer* renderer, int centerX, int centerY, int radius)
{
    int x = radius - 1;
    int y = 0;
    int dx = 1;
    int dy = 1;
    int err = dx - (radius << 1);

    while (x >= y)
    {
        SDL_RenderDrawPoint(renderer, centerX + x, centerY + y);
        SDL_RenderDrawPoint(renderer, centerX + y, centerY + x);
        SDL_RenderDrawPoint(renderer, centerX - y, centerY + x);
        SDL_RenderDrawPoint(renderer, centerX - x, centerY + y);
        SDL_RenderDrawPoint(renderer, centerX - x, centerY - y);
        SDL_RenderDrawPoint(renderer, centerX - y, centerY - x);
        SDL_RenderDrawPoint(renderer, centerX + y, centerY - x);
        SDL_RenderDrawPoint(renderer, centerX + x, centerY - y);

        if (err <= 0) {
            y++;
            err += dy;
            dy += 2;
        }

        if (err > 0) {
            x--;
            dx += 2;
            err += dx - (radius << 1);
        }
    }
}

Eigen::MatrixXd calculate_state(int paddle_x, double puck_x, double puck_y, double puck_dx, double puck_dy)
{
    Eigen::MatrixXd state(1, 6);
    state = Eigen::MatrixXd::Zero(1, 6);

    state(0, 0) = paddle_x + PADDLE_WIDTH / 2 < puck_x ? 1 : 0;
    state(0, 1) = paddle_x + PADDLE_WIDTH / 2 < puck_x ? 0 : 1;
    state(0, 2) = (puck_dx > 0 && puck_dy < 0) ? 1 : 0;
    state(0, 3) = (puck_dx < 0 && puck_dy < 0) ? 1 : 0;
    state(0, 4) = (puck_dx > 0 && puck_dy >= 0) ? 1 : 0;
    state(0, 5) = (puck_dx < 0 && puck_dy >= 0) ? 1 : 0;

    return state;
}

class AirHockey
{
private:
    int Width;
    int Height;
    bool running;
    double puck_x, puck_y, puck_dx, puck_dy;
    int dir_y[2] = { -1, 1 };
    int dir_imp[4] = { -2, -1, 1, 2 };
    int player1_x, player2_x, player1_y, player2_y, player1_score, player2_score, cnt;
    SDL_Renderer* renderer;
public:
    AirHockey(int w = 600, int h = 800) : Width(w), Height(h), running(true)
    {
        SDL_Init(SDL_INIT_EVERYTHING);
        auto window = SDL_CreateWindow("Air Hockey", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);
        renderer = SDL_CreateRenderer(window, -1, 0);

        puck_x = Width / 2;
        puck_y = Height / 2;
        puck_dx = (-1 + 2 * ((float)rand()) / RAND_MAX) * PUCK_SPEED_X;
        puck_dy = dir_y[rand() % 2] * PUCK_SPEED_Y;
        player1_x = Width / 2 - PADDLE_WIDTH / 2;
        player2_x = Width / 2 - PADDLE_WIDTH / 2;
        player1_y = 30;
        player2_y = Height - PADDLE_HEIGHT - player1_y;
        player1_score = 0;
        player2_score = 0;
        cnt = 0;
    }

    double get_puck_x()
    const {
        return puck_x;
    }

    double get_puck_y()
    const {
        return puck_y;
    }

    double get_puck_dx()
    const {
        return puck_dx;
    }

    double get_puck_dy()
    const {
        return puck_dy;
    }

    int get_paddle2_x()
    const {
        return player2_x;
    }

    int get_paddle2_y()
    const {
        return player2_y;
    }

    int get_paddle1_x()
        const {
        return player1_x;
    }

    int get_paddle1_y()
        const {
        return player1_y;
    }

    bool get_done()
    const {
        return !running;
    }

    void reset()
    {
        puck_x = Width / 2;
        puck_y = Height / 2;
        puck_dx = (-1 + 2 * ((float)rand()) / RAND_MAX) * PUCK_SPEED_X;
        puck_dy = dir_y[rand() % 2] * PUCK_SPEED_Y;
        player1_x = Width / 2 - PADDLE_WIDTH / 2;
        player2_x = Width / 2 - PADDLE_WIDTH / 2;
        player1_y = 30;
        player2_y = Height - PADDLE_HEIGHT - player1_y;
        player1_score = 0;
        player2_score = 0;
        running = true;
    }

    pair<pair<int, int>, bool> play_step(int action1, int action2)
    {
        int reward1 = 0;
        int reward2 = 0;
        cnt += 1;
        bool st = false;

        if (cnt == 6000)
        {
            puck_x = Width / 2;
            puck_y = Height / 2;
            puck_dx = (-1 + 2 * ((float)rand()) / RAND_MAX) * PUCK_SPEED_X;
            puck_dy = dir_y[rand() % 2] * PUCK_SPEED_Y;
            cnt = 0;
            player2_score += 1;
            player1_score += 1;
            st = true;
            running = false;
            cout << "out of time\n";
        }

        SDL_Event e;
        if (SDL_PollEvent(&e))
        {
            if (SDL_QUIT == e.type) { running = false; }
        }

        player1_x = player1_x + dir_imp[action2] * PADDLE_SPEED_1;
        player1_x = max(0, min(WIDTH - PADDLE_WIDTH, player1_x));

        player2_x = player2_x + dir_imp[action1] * PADDLE_SPEED_2;
        player2_x = max(0, min(WIDTH - PADDLE_WIDTH, player2_x));

        puck_x += puck_dx;
        puck_y += puck_dy;

        if (puck_x <= 0 || puck_x >= Width)
            puck_dx *= -1;
        if (puck_y <= 0 || puck_y >= Height)
            puck_dy *= -1;

        if (player1_y + PADDLE_HEIGHT >= puck_y - PUCK_RADIUS && player1_x <= puck_x + PUCK_RADIUS && puck_x - PUCK_RADIUS <= player1_x + PADDLE_WIDTH)
        {
            puck_dy *= abs(dir_imp[action2]) == 1 ? -1.05 : -1.1;
            reward1 = 10;
            puck_y = player1_y + PADDLE_HEIGHT > puck_y - PUCK_RADIUS ? player1_y + PADDLE_HEIGHT + PUCK_RADIUS : puck_y;
        }
        if (player2_y <= puck_y + PUCK_RADIUS && player2_y + PADDLE_HEIGHT >= puck_y + PUCK_RADIUS && player2_x <= puck_x + PUCK_RADIUS && puck_x - PUCK_RADIUS <= player2_x + PADDLE_WIDTH)
        {
            puck_dy *= abs(dir_imp[action1]) == 1 ? -1.05 : -1.1;
            reward2 = 10;
            puck_y = player2_y < puck_y + PUCK_RADIUS ? player2_y - PUCK_RADIUS : puck_y;
        }

        update_ui();

        if (puck_y >= Height)
        {
            reward2 = -11;
            reward1 = 1;
            puck_x = Width / 2;
            puck_y = Height / 2;
            puck_dx = (-1 + 2 * ((float)rand()) / RAND_MAX) * PUCK_SPEED_X;
            puck_dy = dir_y[rand() % 2] * PUCK_SPEED_Y;
            player1_score += 1;
            st = true;
        }
        else if (puck_y <= 0)
        {
            reward1 = -11;
            reward2 = 1;
            puck_x = Width / 2;
            puck_y = Height / 2;
            puck_dx = (-1 + 2 * ((float)rand()) / RAND_MAX) * PUCK_SPEED_X;
            puck_dy = dir_y[rand() % 2] * PUCK_SPEED_Y;
            player2_score += 1;
            st = true;
        }

        running = (player2_score == 5 || player1_score == 5) ? false : running;
        cnt = (player2_score == 5 || player1_score == 5) ? 0 : cnt;

        return make_pair(make_pair(reward1, reward2), st);
    }

    void update_ui()
    {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_Rect Player1 = { player1_x, player1_y, PADDLE_WIDTH, PADDLE_HEIGHT };
        SDL_Rect Player2 = { player2_x, player2_y, PADDLE_WIDTH, PADDLE_HEIGHT };
        SDL_RenderFillRect(renderer, &Player1);
        SDL_RenderFillRect(renderer, &Player2);
        DrawCircle(renderer, puck_x, puck_y, PUCK_RADIUS);

        SDL_RenderPresent(renderer);
        SDL_Delay(5);
    }
};

class Agent
{
private:
    double discount_factor;
    double exploration_prob;
    double exploration_decay;
    Sequence model;
    MSE criterion;
    DataSet data;
    Sequence prev_model;
    int prev_reward = 0;
public:
    Agent(Sequence model, double d_f = 0.9, double exp_p = 1.0, double exp_dec = 0.94) : model(model), discount_factor(d_f), exploration_prob(exp_p), exploration_decay(exp_dec), prev_model(model) {}
    Agent(vector<Layer*> arr, double d_f = 0.9, double exp_p = 1.0, double exp_dec = 0.9) : discount_factor(d_f), exploration_prob(exp_p), exploration_decay(exp_dec)
    {
        model = Sequence(arr);
        prev_model = model;
    }

    double get_exp_p()
    const {
        return exploration_prob;
    }

    void set_exp_p(double exp_p)
    {
        exploration_prob = exp_p;
    }

    void save_step(Eigen::MatrixXd& state, int action, int reward, Eigen::MatrixXd& next_state, bool done)
    {
        game_step step{ state, next_state, reward, action, done };
        data.store_data(step);
    }

    void train(const Eigen::MatrixXd& state, int action, int reward, Eigen::MatrixXd& next_state, bool done)
    {
        Eigen::MatrixXd target_q_values = model.forward(next_state);
        double max_target_q_value = target_q_values.maxCoeff();
        Eigen::MatrixXd q_values = model.forward(state);
        double target_q_value = reward + ((1 - done) * (discount_factor * max_target_q_value));
        target_q_values = q_values;
        target_q_values(0, action) = target_q_value;
        target_q_values = target_q_values.transpose();
        criterion.forward(target_q_values, q_values);
        model.backward(criterion.backward(1));

        exploration_prob = done ? exploration_decay * exploration_prob : exploration_prob;
    }

    void train_long_mem(int batch_size)
    {
        vector<game_step> batch = data.get_count() > batch_size ? data.get_batch(batch_size) : data.get_batch(data.get_count());
        int bsize = (data.get_count() > batch_size ? batch_size : data.get_count());
        Eigen::MatrixXd state_0(bsize, 6);
        Eigen::MatrixXd state_1(bsize, 6);
        Eigen::VectorXd reward(bsize, 1);
        Eigen::VectorXi action(bsize, 1);
        Eigen::VectorXd done(bsize, 1);

        vector<game_step>::iterator batch_it = batch.begin();
        for(int i = 0; i < bsize; i++, batch_it++)
        {
            state_0.row(i) = (*batch_it).state;
            state_1.row(i) = (*batch_it).next_state;
            reward(i) = (*batch_it).reward;
            action(i) = (*batch_it).action;
            done(i) = (*batch_it).done;
        }
        
        Eigen::MatrixXd target_q_values = model.forward(state_1);
        Eigen::VectorXd max_target_q_value = target_q_values.rowwise().maxCoeff();
        Eigen::MatrixXd q_values = model.forward(state_0);
        Eigen::VectorXd target_q_value = reward + Eigen::VectorXd((Eigen::VectorXd::Ones(bsize, 1) - done).array() * (discount_factor * max_target_q_value).array());
        target_q_values = q_values;
        for (int i = 0; i < bsize; i++)
        {
            target_q_values(i, action(i, 0)) = target_q_value(i, 0);
        }
        Eigen::MatrixXd target_q_values_t = target_q_values.transpose();
        criterion.forward(target_q_values_t, q_values);
        model.backward(criterion.backward(1));
    }

    int get_action(const Eigen::MatrixXd& state)
    {
        Eigen::MatrixXd::Index idx1, idx2;
        (model.forward(state)).maxCoeff(&idx1, &idx2);
        int action = exploration_prob > 0.2 ? ((double(rand()) / RAND_MAX) < exploration_prob ? rand() % 4 : idx2) : idx2;
        return action;
    }
};

int main(int argc, char* argv[])
{
    AirHockey game;
    Sequence model1;
    model1.append(new Linear(6, 16));
    model1.append(new ReLU());
    model1.append(new Linear(16, 32));
    model1.append(new ReLU());
    model1.append(new Linear(32, 4));
    Agent agent(model1);

    Sequence model2;
    model2.append(new Linear(6, 16));
    model2.append(new ReLU());
    model2.append(new Linear(16, 32));
    model2.append(new ReLU());
    model2.append(new Linear(32, 4));
    Agent agent2(model2);

    for (int i = 0; i < 20; i++)
    {
        int sum_reward1 = 0;
        int sum_reward2 = 0;

        while (!game.get_done())
        {
            Eigen::MatrixXd state = calculate_state(game.get_paddle2_x(), game.get_puck_x(), game.get_puck_y(), game.get_puck_dx(), game.get_puck_dy());
            Eigen::MatrixXd state2 = calculate_state(game.get_paddle1_x(), game.get_puck_x(), game.get_puck_y(), game.get_puck_dx(), game.get_puck_dy());

            int action1 = agent.get_action(state);
            int action2 = agent2.get_action(state2);
            pair<pair<int, int>, bool> reward = game.play_step(action1, action2);
            int reward1 = reward.first.first;
            int reward2 = reward.first.second;
            sum_reward2 += reward1;
            sum_reward1 += reward2;

            Eigen::MatrixXd new_state = calculate_state(game.get_paddle2_x(), game.get_puck_x(), game.get_puck_y(), game.get_puck_dx(), game.get_puck_dy());
            Eigen::MatrixXd new_state2 = calculate_state(game.get_paddle1_x(), game.get_puck_x(), game.get_puck_y(), game.get_puck_dx(), game.get_puck_dy());

            agent.train(state, action1, reward2, new_state, reward.second);
            agent.save_step(state, action1, reward2, new_state, reward.second);
            agent2.train(state2, action2, reward1, new_state2, reward.second);
            agent2.save_step(state2, action2, reward1, new_state2, reward.second);
        }
        game.reset();

        agent.train_long_mem(1000);
        agent2.train_long_mem(1000);
    }

    return 0;
}