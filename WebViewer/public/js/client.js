(function(){

myCanvas = document.getElementById("myCanvas");
myCtx = myCanvas.getContext("2d");
myCanvasImg = myCtx.createImageData(myCanvas.width , myCanvas.height);

window.FE = {
    username:null,
    userid:null,
    socket:null,

    genUID:function(username)
    {
        return username + new Date().getTime() + ""+Math.floor(Math.random()*173+511);
    },

    init:function(username)
    {
        //客户端根据时间和随机数生成uid,以用户名称可以重复，后续改为数据库的UID名称
        this.userid = this.genUID(username);
        this.username = username;

        //链接websocket服务器
        this.socket = io.connect("http://localhost:8080");

        //通知服务器有用户登录
        this.socket.emit("login" , {userid:this.userid , username:this.username});

        //发送一段message
        this.socket.emit("message",{userid:this.userid , username:this.username , content:"first message"});

        // this.socket.on("tick" , function(obj))
        // {

        // }


    },

    userLogOut:function()
    {
        //this.socket.emit("logout",{userid:this.userid , username:this.username , content:"last message"});
        //this.socket.emit("disconnect" , {userid:this.userid , username:this.username});
        location.reload();
    },


    userLogIn:function()
    {
        var username = document.getElementById("username").value;
        
        this.init(username);
    },
}

window.LOGIC = {
    
    drawImg:function()
    {
        for (var i=0;i<myCanvasImg.data.length;i+=4){
            myCanvasImg.data[i]=255;//返回一个对象，其包含指定的 ImageData 对象的图像数据
            myCanvasImg.data[i+1]=255;
            myCanvasImg.data[i+2]=0;
            myCanvasImg.data[i+3]=255;
        }
        myCtx.putImageData(myCanvasImg,0,0);//把图像数据（从指定的 ImageData 对象）放回画布上
    },

}




})()