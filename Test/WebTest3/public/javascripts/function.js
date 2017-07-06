
var fucntion_error="";

"use strict"
function test()
{
    sayHi();
}

"use strict"
function onRoleChanged()
{
    try{

        var radio = document.getElementsByName("role");
        
        for(var i = 0 ; i<radio.length ; ++i){
            if(radio[i].checked){
                if(radio[i].value == "doctor"){
                    document.getElementsByName("username")[0].value="doctor's username or email";
                }
                else{
                    document.getElementsByName("username")[0].value="patient's username or email";
                }
            }
        }
    }
    catch(err){
        fucntion_error = "Error : " + err.message + "\n";
        alert(fucntion_error);
    }
    
}

"use strict"
function sayHi()
{
    alert("hi");
    document.getElementById("test").innerHTML="hello";
}

"use strict"
function onSignIn()
{
    try{
        var pwd = document.forms["form_login"]["password"].value;
        var name = document.forms["form_login"]["username"].value;
        if(pwd == ""){
            throw "password is empty!";
        }
        if(name == ""){
            throw "uername or email is empty!";
        }
    }
    catch(err){
        function_error = "Error : " + err;
        alert(function_error);
    }
    
    
}

