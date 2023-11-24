#include<iostream>
#include<string>
#include<iomanip>
#include<algorithm>
using namespace std;
int main(){
    int n,i;
    cin>>n;
    string str;
    cin>>str;
    for(int i=0;i<str.length();i++){
        if(str[i]!=',')
        {
            cout<<str[i]<<endl;
        }
    }
    return 0;
}