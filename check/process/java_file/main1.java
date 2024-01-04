import java.util.Scanner;
public class Main{
    public static void main(String[] args){
        int num;
        Scanner sc = new Scanner(System.in);
        num = sc.nextInt();
        if(num>0){
            if(Tool.isPower(num)){
                System.out.println("yes");
            }
            else{
                System.out.println("no");
            }
        }
        else{
            System.out.println("error");
        }
    }
}
class Tool{
    public static boolean isPower(int num){
        int i=0;
        int n=2;
        while(n<num){
            n=n*2;
            i++;
        }
        if(n==num)
            return true;
        else
            return false;
    }
}