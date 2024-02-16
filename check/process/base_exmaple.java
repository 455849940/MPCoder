public class Student {
    private int id;
    private String name;
    private String gender;
    private String major;
    private int score;

    public Student(int id, String name, String gender, String major, int score) {
        this.id = id;
        this.name = name;
        this.gender = gender;
        this.major = major;
        this.score = score;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getMajor() {
        return major;
    }

    public void setMajor(String major) {
        this.major = major;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    public abstract String getGrade();
}

public class Undergraduate extends Student {
    public Undergraduate(int id, String name, String gender, String major, int score) {
        super(id, name, gender, major, score);
    }

    @Override
    public String getGrade() {
        if (score >= 80 && score <= 100) {
            return "A";
        } else if (score >= 70 && score <= 80) {
            return "B";
        } else if (score >= 60 && score <= 70) {
            return "C";
        } else if (score >= 50 && score <= 60) {
            return "D";
        } else {
            return "E";
        }
    }
}

public class Graduate extends Student {
    private String supervisor;

    public Graduate(int id, String name, String gender, String major, String supervisor, int score) {
        super(id, name, gender, major, score);
        this.supervisor = supervisor;
    }

    @Override
    public String getGrade() {
        if (score >= 90 && score <= 100) {
            return "A";
        } else if (score >= 80 && score <= 90) {
            return "B";
        } else if (score >= 70 && score <= 80) {
            return "C";
        } else if (score >= 60 && score <= 70) {
            return "D";
        } else {
            return "E";
        }
    }
}

public class Main {
    public static void main(String[] args) {
        Undergraduate undergraduate = new Undergraduate(2, "chen", "female", "cs", 90);
        Graduate graduate = new Graduate(3, "li", "male", "sc", "wang", 80);

        System.out.println(undergraduate.getGrade());
        System.out.println(graduate.getGrade());
    }
}