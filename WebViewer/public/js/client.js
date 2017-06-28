(function(){
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


    },

    userLogin:function()
    {
        var username = document.getElementById("username").value;
        
        this.init(username);
    },

    userLogOut:function()
    {
        location.reload();
    }
}
})()