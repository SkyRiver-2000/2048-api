#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<cmath>
using namespace std;

constexpr int NUM_LOG = 15000;

int to_int(const string& s)
{
    int i, ans = 0, len = s.length(), temp = 1;
    for (i = 1; i <= len; ++i)
    {
        ans += (s[len - i] - '0') * temp;
        temp *= 10;
    }
    return ans;
}

double to_double(const string& s)
{
    double ans = 100 * (s[1] - '0') + 10 * (s[2] - '0') + (s[3] - '0') + 0.1 * (s[5] - '0');
    if (s[0] <= '9' && s[0] >= '0') ans += 1000 * (s[0] - '0');
    return ans;
}

int main()
{
    int i, j, pos, more_than_1024 = 0, freq[12] = {0};
    double max_score = 0, temp, sum = 0;
    ifstream fin;
    ofstream fout("./score.csv");
    fout << "Score" << endl;
    string line, s;
    for (i = 1; i <= NUM_LOG; ++i)
    {
        // Open a log with pre-defined naming rule: tempX.log
        // Read the log line by line
        if (i % 1000 == 0) cout << setw(5) << right << i << " logs have been processed!" << endl;
        fin.open(string("log_storage/temp") + to_string(i) + string(".log"));
        while (getline(fin, line))
        {
            // Find the end of a game, format: You lose(win)! Score: XXXX
            // Compute the average score per game
            // Compute the probability of reaching 2048
            if (line[0] == 'Y' && line[1] == 'o')
            {
                // The score of a game must be a 3 or 4 bit integer
                j = line.length();
                while(line[j - 1] >= '0' && line[j - 1] <= '9') --j;
                s = line.substr(j, line.length() - j);
                temp = to_int(s);
                // Compute average score per game and 2048 probability
                sum += temp;
                ++freq[int(log2(temp))];
                fout << temp << endl;
            }
            // Find the end of a log, format: Average scores: @10 times XXXX.X
            // Look for the best log in the whole simulation
            // Compute the probability of obtaining 1024 and higher score
            if (line[0] == 'A' && line[1] == 'v')
            {
                // Average score must be a 5 or 6 bit float with a bit after dot
                s = line.substr(line.length() - 6, 6);
                temp = to_double(s);
                // Compute probability
                if (temp >= 1024) ++more_than_1024;
                // Look for the best log
                if (max_score < temp)
                {
                    max_score = temp;
                    pos = i;
                }
            }
        }
        fin.close();
    }
    // Print necessary information:
    // Average score per game, best log and its score, the probability of 1024 and higher
    cout << endl << "Average score per game: " << sum / (NUM_LOG * 10) << endl;
    cout << "Max average score @10 times: " << max_score << endl << "It appears at: temp" << pos << ".log" << endl;
    cout << "The proportion of 1024 and higher average score is: " << double(more_than_1024) / NUM_LOG * 100 << '%' << endl;
    cout << "The proportion of victory is: " << double(freq[11]) / (NUM_LOG * 10) * 100 << '%' << endl;
    cout << "Statistics:" << endl << "Tile" << '\t';
    for (i = 1; i <= 11; ++i)
        cout << pow(2, i) << '\t';
    cout << endl << "Time" << '\t';
    for (i = 1; i <= 11; ++i)
        cout << freq[i] << '\t';
    cout << endl;

    // Write statistics into .csv file for visualization in python
    fout.close();
    fout.open("./stats.csv");
    fout << "Max_Tile,Frequence" << endl;
    for (i = 1; i <= 11; ++i)
        fout << pow(2, i) << ',' << freq[i] << endl;
    fout.close();

    return 0;
}