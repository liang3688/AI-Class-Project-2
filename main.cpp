#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>

using namespace std;

bool sortcol(const vector<double>& v1, const vector<double>& v2){
    return v1[0] < v2[0];
}

unsigned Most_common_class_count(vector<unsigned int> label){
    unsigned int max_count = 0;
    for(unsigned int i = 0; i < label.size(); i++){
        unsigned int current_count = 1;
        for(unsigned int j = i+1; j < label.size(); j++){
            if(label[i] == label[j])
                current_count++;
        }
        if(current_count > max_count)
            max_count = current_count;
    }
    return max_count;
}

class Classifier {

public:
    unsigned int Test(vector<vector<double>> instances, vector<unsigned int> label, unsigned int instanceID){
        vector<vector<double>> table;
        unsigned int predicted_label;
        for(unsigned int i = 0; i < instances.size(); i++){
            if(instanceID != i){
                table.push_back(vector<double>());
                table[table.size()-1].push_back(Euclidean(instances[instanceID], instances[i]));
                table[table.size()-1].push_back(label[i]);
            }
        }
        sort(table.begin(), table.end(), sortcol);
        predicted_label = static_cast<unsigned int>(table[0][1]);
        return predicted_label;
    }

    double Euclidean(vector<double> instance1, vector<double> instance2){
        double distance = 0;
        for(unsigned int i = 0; i < instance1.size(); i++)
            distance += pow(instance1[i]-instance2[i], 2);
        distance = sqrt(distance);
        return distance;
    }

};

class Validator {
private:
    vector<vector<double>> feature_instance;
    vector<unsigned int> label;

public:

    void Read_data(string file_name){
        ifstream infile;
        infile.open(file_name);
        if(!infile.is_open()){
            cout << "\nCannot open the file \"" << file_name << "\".\n";
            exit(1);
        }

        string buffer;
        unsigned int i = 0;
        double temp;
        while(getline(infile, buffer)){
            stringstream line(buffer);

            if(line >> temp)
                label.push_back(static_cast<unsigned int>(temp));

            feature_instance.push_back(vector<double>());
            while(line >> temp)
                feature_instance[i].push_back(static_cast<double>(temp));
            i++;
        }
        cout << "\nThis dataset has " << feature_instance[0].size() <<
                " features (not inculding the class attribute), with " << feature_instance.size()
             << " instances\n";
    }

    void Normalize(){
        cout << "\nPlease wait while I normalize the data...\t";
        double mean, sd, var, sum;
        for(unsigned int i = 0; i < feature_instance[0].size(); i++){
            mean = 0;
            sd = 0;
            var = 0;
            sum = 0;
            for(unsigned int j = 0; j < feature_instance.size(); j++){
                sum += feature_instance[j][i];
                mean = sum/feature_instance.size();
            }

            for(unsigned int j = 0; j < feature_instance.size(); j++)
                var += pow(feature_instance[j][i] - mean, 2);

            var = var/feature_instance.size();
            sd = sqrt(var);

            for(unsigned int j = 0; j < feature_instance.size(); j++)
                feature_instance[j][i] = (feature_instance[j][i] - mean)/sd;
        }
        cout << "Done!\n";
    }

    double LOOV(vector<unsigned int> feature_subset){
        if(feature_subset.empty())
            return static_cast<double>(Most_common_class_count(label))/static_cast<double>(label.size());

        unsigned int correct_count = 0;
        double accuracy = 0;
        vector<vector<double>> subset;
        Classifier c;

        for(unsigned int i = 0; i < feature_subset.size(); i++)
            feature_subset[i]--;

        for(unsigned int i = 0; i < feature_instance.size(); i++){
            subset.push_back(vector<double>());
        }

        for(unsigned int i = 0; i < feature_subset.size(); i++){
            unsigned int col = feature_subset[i];
            for(unsigned int j = 0; j < feature_instance.size(); j++){
                subset[j].push_back(feature_instance[j].at(col));
            }
        }

        for(unsigned int i = 0; i < label.size(); i++){
            if(c.Test(subset, label, i) == label[i])
                correct_count++;
        }
        accuracy = static_cast<double>(correct_count)/static_cast<double>(label.size());
        return accuracy;
    }

    unsigned int instance_size(){
        return feature_instance.size();
    }

    unsigned int feature_count(){
        return feature_instance[0].size();
    }

};

class Feature
{
private:
    double accuracy;
    vector<unsigned int> feature_list;

public:
    Feature(){}

    Feature(Validator v){
        accuracy = v.LOOV(feature_list) * 100;
    }

    vector<unsigned int> get_feature_list() const {
        return feature_list;
    }

    double evaluate(){
        return accuracy;
    }

    bool add_new_feature(unsigned int num, Validator v){
        for(unsigned int i = 0; i < feature_list.size(); i++){
            if(feature_list[i] == num)
                return false;
        }
        feature_list.push_back(num);
        accuracy = v.LOOV(feature_list) * 100;
        return true;
    }

    bool delete_feature(unsigned int num, Validator v){
        for(unsigned int i = 0; i < feature_list.size(); i++){
            if(feature_list[i] == num){
                feature_list.erase(remove(feature_list.begin(), feature_list.end(), num), feature_list.end());
                accuracy = v.LOOV(feature_list) * 100;
                return true;
            }
        }
        return false;
    }

    void print_feature() const {
        cout << "{";
        if(feature_list.empty())
            cout << " ";
        else{
            for(unsigned int i = 0; i < feature_list.size(); i++){
                cout << feature_list[i];
                if(i < feature_list.size()-1)
                    cout << ",";
            }
        }
        cout << "}";
    }

};

void Forward(Validator v){
    Feature f(v);
    Feature max;
    Feature temp;
    unsigned int num = v.feature_count();
    bool new_max = true;
    cout << "\nRunning nearest neighbor with no features(default rate), using \"leaving-one-out\" evaluation, "
            "I get an accuracy of " << f.evaluate() << "%\n";
    max = f;
    temp = f;
    cout << "\nBeginning search.\n\n";

    for(unsigned int i = 0; i < num && new_max; i++){
        new_max = false;
        for(unsigned int j = 1; j <= num; j++){
            f = temp;
            if(f.add_new_feature(j, v)){
                cout << "Using feature(s) ";
                f.print_feature();
                cout << " accuracy is " << f.evaluate() << "%\n";
                if(f.evaluate() > max.evaluate()){
                    max = f;
                    new_max = true;
                }
            }
        }
        if(new_max){
            temp = max;
            cout << "\nFeature set ";
            max.print_feature();
            cout << " was the best, accuracy is " << max.evaluate() << "%\n\n";
        }
        else
            cout << "\n(Warning, Accuracy has decreased!)\n\n";
    }
    cout << "Finished search!! The best feature subset is ";
    max.print_feature();
    cout << ", which has an accuracy of " << max.evaluate() << "%\n";
}

void Backward(Validator v){
    Feature f;
    Feature max;
    Feature temp;
    unsigned int num = v.feature_count();
    bool new_max = true;
    for(unsigned int j = 1; j <= num; j++){
        f.add_new_feature(j, v);
    }
    cout << "\nUsing feature(s) ";
    f.print_feature();
    cout << setprecision(1) << fixed << " accuracy is " << f.evaluate() << "%\n\n";
    cout << "Beginning search.\n\n";
    max = f;
    temp = f;
    for(unsigned int i = 0; i < num && new_max; i++){
        new_max = false;
        for(unsigned int j = 1; j <= v.feature_count(); j++){
            f = temp;
            if(f.delete_feature(j, v)){
                cout << "Using feature(s) ";
                f.print_feature();
                cout << " accuracy is " << f.evaluate() << "%\n";
                if(f.evaluate() > max.evaluate()){
                    max = f;
                    new_max = true;
                }
            }
        }
        if(new_max){
            temp = max;
            cout << "\nFeature set ";
            max.print_feature();
            cout << " was the best, accuracy is " << max.evaluate() << "%\n\n";
        }
        else
            cout << "\n(Warning, Accuracy has decreased!)\n\n";
    }
    cout << "Finished search!! The best feature subset is ";
    max.print_feature();
    cout << ", which has an accuracy of " << max.evaluate() << "%\n";
}

int main()
{
    Validator val;
    string fn;
    int choice;
    cout << "Welcome to Yongfeng Liang's Feature Selection Algorithm.\n\n"; 
    cout << "Type in the name of the file to test: ";
    cin >> fn;
    val.Read_data(fn);
    val.Normalize();
    cout << "\n\nType the number of the algorithm you want to run.\n\n"
            "       *Forward Selection\n"
            "       *Backward Elimination\n\n\t\t\t";
    cin >> choice;
    if(choice == 1)
        Forward(val);
    else if(choice == 2)
        Backward(val);
    else
        cout << "\nInvalid Input, Program terminated.\n";
    return 0;
}
