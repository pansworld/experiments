//
//  MicrosoftViewController.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/30/23.
//

import SwiftUI
import MSAL
import Foundation


//let helper = Helpers()

//let kClientID = "3de9c462-dfd5-4082-a1af-b0e62822b8ab"
//let kRedirectUri = "msauth.com.microsoft.identitysample.MSALiOS://auth"
//let kAuthority = "https://login.microsoftonline.com/common"
//let kGraphEndpoint = "https://graph.microsoft.com/"

class MicrosoftViewController:UIViewController{
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    var accessToken: String = ""
    private var application :Any
    private var accountIdentifier :String = ""
    @Binding var elProb: [String]
    @Binding var weekdayData: [WeekdayData]

    init(value: Binding<[String]>, data: Binding<[WeekdayData]>) {
        do {
            self._elProb = value
            self._weekdayData = data
            let authority = try MSALAuthority(url: URL(string: helper.msalConfig.authorityURL)!)
            
            let pcaConfig = MSALPublicClientApplicationConfig(clientId: helper.msalConfig.clientId, redirectUri: helper.msalConfig.redirectURI, authority: authority)
            application = try MSALPublicClientApplication(configuration: pcaConfig)
            super.init(nibName: nil, bundle: nil)
        }catch let error as NSError{
            application = ""
            super.init(nibName: nil, bundle: nil)
            print(error.localizedDescription)
        }

    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        let btn = UIButton(frame: CGRect(x:20, y: self.view.frame.height - 100, width: self.view.frame.width - 40, height: 52))
        btn.backgroundColor = .systemBlue
        btn.setTitle("Sign In", for: .normal)
        btn.setTitleColor(.white, for: .normal)
        btn.addTarget(self, action: #selector(buttonTapped), for: .touchUpInside)
        
        self.view.addSubview(btn)
    }
    
    @objc func buttonTapped(_ sender: UIButton) {
        /*do {*/
            var isSignIn = false
            if (sender.currentTitle == "Sign In"){
                isSignIn = true

            }else{
                isSignIn = false
            }
            
            if (!isSignIn){
                print("Sign out")
                Task {
                    let webViewParameters = MSALWebviewParameters(authPresentationViewController: self)
                    let signoutParameters = MSALSignoutParameters(webviewParameters: webViewParameters)
                    signoutParameters.signoutFromBrowser = false

                    
                    let account = try (self.application as! MSALPublicClientApplication).account(forIdentifier: self.accountIdentifier )
                    print(account)
                    try await (self.application as!
                         MSALPublicClientApplication).signout(with: account, signoutParameters: signoutParameters)
                    
                    sender.setTitle("Sign In", for: .normal)
                    sender.backgroundColor = .blue
                    
                    self.elProb[0]="0.00"
                    self.elProb[1]="0.00"
                    self.elProb[2]="0.00"
                    self.weekdayData=[]
                    
                }

            }else{
                
                let webViewParameters = MSALWebviewParameters(authPresentationViewController: self)
                let interactiveParameters = MSALInteractiveTokenParameters(scopes: ["user.read","Files.ReadWrite.All"], webviewParameters: webViewParameters)
                interactiveParameters.promptType = .selectAccount
                (self.application as! MSALPublicClientApplication).acquireToken(with: interactiveParameters){ (result, error) in
                    guard let result = result else {
                        print(error!)
                        return
                    }
                    helper.accessToken = result.accessToken
                    print(result.tenantProfile.claims!["name"]!)

                    sender.setTitle("Sign Out", for: .normal)
                    sender.backgroundColor = .red

                    Task {
                        await helper.getLastExcelTableRow(excelFile: "elevator_experiments.xlsx", tableName: "Table1")
                        //print("Helper=\(helper.tableRow)")
                        //print(helper.tableRow[0])
                        helper.el1Prob = String(format: "%.2f", helper.tableRow[0][13] as! Double)
                        helper.el2Prob = String(format: "%.2f", helper.tableRow[0][14] as! Double)
                        helper.el3Prob = String(format: "%.2f", helper.tableRow[0][15] as! Double)
                        self.elProb[0]=helper.el1Prob
                        self.elProb[1]=helper.el2Prob
                        self.elProb[2]=helper.el3Prob
                        self.weekdayData=helper.weekdayData
                    }
                }
            }
        /*}catch{
            self.accessToken = ""
            print(error.localizedDescription)
            return
        }*/
        

    }
}

struct MyCustomLogin: UIViewControllerRepresentable{
    
    @Binding var elProb: [String]
    @Binding var weekdayData: [WeekdayData]

    init(value: Binding<[String]>, data: Binding<[WeekdayData]>){
        self._elProb = value
        self._weekdayData = data
    }
    
    func updateUIViewController(_ uiViewController: MicrosoftViewController, context: Context) {
    }
    
    typealias UIViewControllerType = MicrosoftViewController
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<MyCustomLogin>) -> MicrosoftViewController {
        return MicrosoftViewController(value: self._elProb, data: self._weekdayData)
    }
}
