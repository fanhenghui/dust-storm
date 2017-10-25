var alreadyLoggedUsers = [];

module.exports = {
    addLoggedUser: function (uname) {
        return alreadyLoggedUsers.push(uname);
    },
    includesUser: function (uname) {
        return alreadyLoggedUsers.includes(uname);
    },
    removeLoggedUser: function (uname) {
        return alreadyLoggedUsers.splice(alreadyLoggedUsers.indexOf(uname), 1);
    }
};