function Hello(){
    var name;
    this.setName = function(n){
        name = n;
    }
    this.sayName = function(){
        console.log("my name is " , name);
    }
}

module.exports = Hello;